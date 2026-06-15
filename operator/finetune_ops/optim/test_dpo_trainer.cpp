#include "dpo_trainer.h"
#include "../graph/auto_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

fs::path make_case_dir() {
    const fs::path root = fs::temp_directory_path() / "mft_dpo_trainer";
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

ops::AutoModelLoadOptions no_weight_load() {
    ops::AutoModelLoadOptions options;
    options.load_weights = false;
    options.verbose = false;
    return options;
}

void fill_parameters(ops::AutoModelForCausalLM& model) {
    int offset = 0;
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = 0.01f + 0.001f * static_cast<float>((offset + i) % 17);
        }
        offset += static_cast<int>(param->numel());
    }
}

ops::AutoLoraConfig tiny_lora() {
    ops::AutoLoraConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 4.0f;
    cfg.dropout = 0.0f;
    cfg.seed = 11;
    return cfg;
}

ops::PreferenceBatch make_cached_ref_batch() {
    ops::PreferenceBatch batch;
    const std::vector<int64_t> shape{1, 4};
    const int32_t chosen_ids[] = {1, 2, 3, 0};
    const int32_t rejected_ids[] = {1, 4, 5, 0};
    const float attention[] = {1.0f, 1.0f, 1.0f, 0.0f};
    const int32_t response_mask[] = {0, 1, 1, 0};

    batch.chosen_input_ids = std::make_shared<ops::Tensor>(shape, chosen_ids, ops::kInt32, ops::kCPU);
    batch.rejected_input_ids = std::make_shared<ops::Tensor>(shape, rejected_ids, ops::kInt32, ops::kCPU);
    batch.chosen_attention_mask = std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.rejected_attention_mask = std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.chosen_response_mask = std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.rejected_response_mask = std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.ref_chosen_logps = {-2.0f};
    batch.ref_rejected_logps = {-2.1f};
    batch.sample_indices = {0};
    batch.batch_size = 1;
    batch.sequence_length = 4;
    batch.valid_response_token_count = 4;
    return batch;
}

float max_param_delta(const std::vector<ops::TensorPtr>& before,
                      const std::vector<ops::TensorPtr>& after) {
    require(before.size() == after.size(), "parameter vector size mismatch");
    float max_delta = 0.0f;
    for (std::size_t i = 0; i < before.size(); ++i) {
        const float* a = before[i]->data<float>();
        const float* b = after[i]->data<float>();
        for (int64_t j = 0; j < before[i]->numel(); ++j) {
            max_delta = std::max(max_delta, std::fabs(a[j] - b[j]));
        }
    }
    return max_delta;
}

class FixedPreferenceProvider final : public ops::PreferenceBatchProvider {
public:
    explicit FixedPreferenceProvider(ops::PreferenceBatch batch)
        : batch_(std::move(batch)) {}

    ops::PreferenceBatch next_batch(std::size_t, bool loop) override {
        if (used_ && !loop) {
            return ops::PreferenceBatch{};
        }
        used_ = true;
        return batch_;
    }

    void reset() override { used_ = false; }
    std::size_t num_pairs() const override { return 1; }

private:
    ops::PreferenceBatch batch_;
    bool used_ = false;
};

void test_dpo_trainer_updates_lora() {
    const fs::path root = make_case_dir();
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    fill_parameters(*model);
    model->init_lora(tiny_lora());

    std::vector<ops::TensorPtr> before;
    for (const auto& param : model->trainable_parameters()) {
        before.push_back(param->clone());
    }
    require(!before.empty(), "DPOTrainer test missing LoRA parameters");

    ops::DPOTrainerConfig cfg;
    cfg.learning_rate = 1e-2f;
    cfg.beta = 0.1f;
    cfg.use_streaming_dpo_loss = true;
    ops::DPOTrainer trainer(*model, cfg);
    const auto result = trainer.train_step(make_cached_ref_batch());

    require(std::isfinite(result.loss), "DPOTrainer loss is not finite");
    require(result.trainable_tensor_count > 0, "DPOTrainer trainable count missing");
    require(result.pair_count == 1, "DPOTrainer pair count mismatch");
    require(result.optimizer_step, "DPOTrainer default step should update optimizer");
    require(max_param_delta(before, model->trainable_parameters()) > 0.0f,
            "DPOTrainer did not update LoRA parameters");

    fs::remove_all(root);
}

void test_dpo_fit_provider() {
    const fs::path root = make_case_dir();
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    fill_parameters(*model);
    model->init_lora(tiny_lora());

    ops::DPOTrainerConfig cfg;
    cfg.learning_rate = 1e-2f;
    cfg.beta = 0.1f;
    ops::DPOTrainer trainer(*model, cfg);
    FixedPreferenceProvider provider(make_cached_ref_batch());

    ops::DPOFitConfig fit;
    fit.max_steps = 1;
    fit.batch_size = 1;
    fit.loop_dataset = false;
    bool callback_called = false;
    auto result = trainer.fit(provider, fit, [&](const ops::DPOFitStep& step) {
        callback_called = true;
        require(step.step == 1, "DPO fit callback step mismatch");
        require(std::isfinite(step.train_result.loss), "DPO fit callback loss is not finite");
    });

    require(callback_called, "DPO fit callback was not called");
    require(result.completed_steps == 1, "DPO fit completed step mismatch");
    require(!result.stopped_early, "DPO fit stopped early unexpectedly");
    require(std::isfinite(result.mean_loss), "DPO fit mean loss is not finite");

    fs::remove_all(root);
}

}  // namespace

int main() {
    test_dpo_trainer_updates_lora();
    test_dpo_fit_provider();
    return 0;
}
