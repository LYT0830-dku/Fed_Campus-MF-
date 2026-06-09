#include "auto_model.h"
#include "../optim/auto_trainer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

fs::path make_case_dir(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("mft_auto_model_" + name);
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

ops::AutoLoraConfig tiny_lora() {
    ops::AutoLoraConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 4.0f;
    cfg.dropout = 0.0f;
    cfg.seed = 7;
    return cfg;
}

void fill_parameters(ops::AutoModelForCausalLM& model, float value) {
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = value;
        }
    }
}

void require_named_trainables_are_stable(ops::AutoModelForCausalLM& model,
                                         const std::string& expected_name_fragment) {
    const auto params = model.trainable_parameters();
    const auto named = model.named_trainable_parameters();
    require(named.size() == params.size(), "named trainable count mismatch");
    require(!named.empty(), "named trainable parameter list is empty");

    std::unordered_set<std::string> seen;
    bool saw_expected_fragment = false;
    for (const auto& kv : named) {
        require(!kv.first.empty(), "named trainable parameter has empty name");
        require(kv.second != nullptr, "named trainable parameter tensor is null");
        require(seen.insert(kv.first).second, "duplicate named trainable parameter: " + kv.first);
        if (kv.first.find(expected_name_fragment) != std::string::npos) {
            saw_expected_fragment = true;
        }
    }
    require(saw_expected_fragment,
            "named trainable parameters missing expected fragment: " + expected_name_fragment);
}

class FixedBatchProvider final : public ops::BatchProvider {
public:
    explicit FixedBatchProvider(std::vector<ops::CausalLMBatch> batches)
        : batches_(std::move(batches)) {}

    ops::CausalLMBatch next_batch(std::size_t, bool loop) override {
        if (batches_.empty()) {
            return ops::CausalLMBatch{};
        }
        if (cursor_ >= batches_.size()) {
            if (!loop) {
                return ops::CausalLMBatch{};
            }
            cursor_ = 0;
        }
        return batches_[cursor_++];
    }

    void reset() override {
        cursor_ = 0;
    }

    std::size_t num_sequences() const override {
        return batches_.size();
    }

private:
    std::vector<ops::CausalLMBatch> batches_;
    std::size_t cursor_ = 0;
};

void test_gpt2_auto_trainer_step() {
    const fs::path root = make_case_dir("gpt2");
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::GPT2, "GPT-2 family dispatch failed");
    fill_parameters(*model, 0.01f);

    model->init_lora(tiny_lora());
    require(!model->trainable_parameters().empty(), "GPT-2 LoRA parameters missing");
    require(model->trainable_parameters().size() == 4,
            "GPT-2 default LoRA should use fused c_attn plus attention projection");
    require_named_trainables_are_stable(*model, "blocks.0.attn.qkv");

    int32_t ids[] = {1, 2, 3};
    float mask[] = {1.0f, 1.0f, 1.0f};
    int32_t labels[] = {-100, 2, 3};
    auto input_ids = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, ids, ops::kInt32, ops::kCPU);
    auto attention_mask = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, mask, ops::kFloat32, ops::kCPU);
    auto label_tensor = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 3}, labels, ops::kInt32, ops::kCPU);

    ops::AutoTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-3f;
    ops::AutoTrainer trainer(*model, trainer_cfg);
    const auto result = trainer.train_step(input_ids, attention_mask, label_tensor);
    require(std::isfinite(result.loss), "GPT-2 AutoTrainer loss is not finite");
    require(result.trainable_tensor_count > 0, "GPT-2 trainable tensor count missing");
    require(result.valid_label_count == 2, "GPT-2 valid label count mismatch");
    require(result.optimizer_step, "GPT-2 AutoTrainer default step should update optimizer");
    require(result.accumulation_step == 1, "GPT-2 default accumulation step mismatch");
    require(result.gradient_accumulation_steps == 1, "GPT-2 default grad accumulation mismatch");

    ops::AutoTrainerConfig accum_trainer_cfg;
    accum_trainer_cfg.learning_rate = 1e-3f;
    accum_trainer_cfg.gradient_accumulation_steps = 2;
    ops::AutoTrainer accum_trainer(*model, accum_trainer_cfg);
    const auto accum_first = accum_trainer.train_step(input_ids, attention_mask, label_tensor);
    require(std::isfinite(accum_first.loss), "GPT-2 accumulated first loss is not finite");
    require(!accum_first.optimizer_step, "GPT-2 first accumulated step should not update optimizer");
    require(accum_first.accumulation_step == 1, "GPT-2 first accumulated step counter mismatch");
    require(accum_first.gradient_accumulation_steps == 2, "GPT-2 accumulation config mismatch");
    const auto accum_second = accum_trainer.train_step(input_ids, attention_mask, label_tensor);
    require(std::isfinite(accum_second.loss), "GPT-2 accumulated second loss is not finite");
    require(accum_second.optimizer_step, "GPT-2 second accumulated step should update optimizer");
    require(accum_second.accumulation_step == 2, "GPT-2 second accumulated step counter mismatch");
    require(std::isfinite(accum_second.accumulated_loss), "GPT-2 accumulated average loss is not finite");

    ops::CausalLMBatchConfig batch_cfg;
    batch_cfg.sequence_length = 3;
    auto batch = ops::make_causal_lm_batch_from_token_ids({{1, 2, 3}}, 0, batch_cfg);
    const auto batch_result = trainer.train_step(batch);
    require(std::isfinite(batch_result.loss), "GPT-2 AutoTrainer batch loss is not finite");
    require(batch_result.valid_label_count == 2, "GPT-2 batch valid label count mismatch");

    FixedBatchProvider provider({
        ops::make_causal_lm_batch_from_token_ids({{1, 2, 3}}, 0, batch_cfg),
        ops::make_causal_lm_batch_from_token_ids({{2, 3, 4}}, 0, batch_cfg),
    });
    ops::AutoFitConfig fit_cfg;
    fit_cfg.max_steps = 2;
    fit_cfg.batch_size = 1;
    fit_cfg.loop_dataset = false;
    int callbacks = 0;
    const auto fit_result = trainer.fit(provider, fit_cfg, [&](const ops::AutoFitStep& step) {
        ++callbacks;
        require(step.step == callbacks, "AutoTrainer fit callback step mismatch");
        require(std::isfinite(step.train_result.loss), "AutoTrainer fit callback loss is not finite");
    });
    require(fit_result.completed_steps == 2, "AutoTrainer fit completed step count mismatch");
    require(!fit_result.stopped_early, "AutoTrainer fit stopped early unexpectedly");
    require(callbacks == 2, "AutoTrainer fit callback count mismatch");
    require(fit_result.total_valid_label_count == 4, "AutoTrainer fit valid label count mismatch");
    require(std::isfinite(fit_result.final_loss), "AutoTrainer fit final loss is not finite");
    require(std::isfinite(fit_result.mean_loss), "AutoTrainer fit mean loss is not finite");
}

void test_gpt2_lora_target_policy() {
    const fs::path root = make_case_dir("gpt2_lora_policy");
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto fused = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    fused->init_lora(tiny_lora());
    require(fused->trainable_parameters().size() == 4,
            "GPT-2 default LoRA policy must be PEFT-style fused c_attn + attn projection");

    ops::AutoLoraConfig split_cfg = tiny_lora();
    split_cfg.target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"};
    auto split = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    split->init_lora(split_cfg);
    require(split->trainable_parameters().size() == 8,
            "GPT-2 explicit q/k/v/o policy must use split qkv LoRA slices");

    ops::AutoLoraConfig peft_cfg = tiny_lora();
    peft_cfg.target_modules = {"c_attn", "c_proj"};
    auto peft = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    peft->init_lora(peft_cfg);
    require(peft->trainable_parameters().size() == 6,
            "GPT-2 PEFT c_attn+c_proj should include fused qkv, attention proj, and mlp c_proj");
    require_named_trainables_are_stable(*peft, "blocks.0.mlp.fc_out");

    ops::AutoLoraConfig ambiguous_cfg = tiny_lora();
    ambiguous_cfg.target_modules = {"c_attn", "q_proj"};
    auto ambiguous = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    bool threw = false;
    try {
        ambiguous->init_lora(ambiguous_cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("ambiguous") != std::string::npos;
    }
    require(threw, "GPT-2 mixed fused/split qkv targets should be rejected");
}

void test_gemma_auto_lora_dispatch() {
    const fs::path root = make_case_dir("gemma");
    write_file(root / "config.json",
               R"({"model_type":"gemma3_text","vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"sliding_window":8})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::Gemma, "Gemma family dispatch failed");
    model->init_lora(tiny_lora());
    require(!model->trainable_parameters().empty(), "Gemma LoRA parameters missing");
    require_named_trainables_are_stable(*model, "layers.0.self_attn.q_proj");
}

void test_qwen_auto_lora_dispatch() {
    const fs::path root = make_case_dir("qwen");
    write_file(root / "config.json",
               R"({"model_type":"qwen2","vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::Qwen, "Qwen family dispatch failed");
    model->init_lora(ops::AutoLoraConfig::attention_qkvo());
    require(!model->trainable_parameters().empty(), "Qwen LoRA parameters missing");
    require_named_trainables_are_stable(*model, "layers.0.self_attn.q_proj");
}

void test_llama_auto_trainer_step_with_untied_head() {
    const fs::path root = make_case_dir("llama");
    write_file(root / "config.json",
               R"({"model_type":"llama","tie_word_embeddings":false,"vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":16,"rms_norm_eps":0.000001,"rope_theta":10000.0})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    require(model->family() == ops::ModelFamily::Llama, "Llama family dispatch failed");
    require(model->llama() != nullptr, "Llama concrete model missing");
    fill_parameters(*model, 0.01f);

    model->init_lora(ops::AutoLoraConfig::attention_qkvo());
    require(!model->trainable_parameters().empty(), "Llama LoRA parameters missing");
    require(model->trainable_parameters().size() == 8, "Llama q/k/v/o LoRA param count mismatch");
    require_named_trainables_are_stable(*model, "layers.0.self_attn.q_proj");

    int32_t ids[] = {1, 2, 3, 4};
    float mask[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int32_t labels[] = {-100, 2, 3, 4};
    auto input_ids = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 4}, ids, ops::kInt32, ops::kCPU);
    auto attention_mask = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 4}, mask, ops::kFloat32, ops::kCPU);
    auto label_tensor = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, 4}, labels, ops::kInt32, ops::kCPU);

    ops::AutoTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-3f;
    ops::AutoTrainer trainer(*model, trainer_cfg);
    const auto result = trainer.train_step(input_ids, attention_mask, label_tensor);
    require(std::isfinite(result.loss), "Llama AutoTrainer loss is not finite");
    require(result.trainable_tensor_count == 8, "Llama trainable tensor count mismatch");
    require(result.valid_label_count == 3, "Llama valid label count mismatch");
}

void test_llama_key_mapper_respects_bias_and_lm_head() {
    const auto no_bias = ops::LlamaKeyMapper::generate_llama_mapping(1, false, false);
    require(no_bias.count("lm_head.weight") == 1, "Llama untied mapper must include lm_head.weight");
    require(no_bias.count("layers.0.self_attn.q_proj.bias") == 0,
            "Llama no-bias mapper should not require q_proj.bias");
    require(no_bias.at("layers.0.input_norm.weight") ==
                "model.layers.0.input_layernorm.weight",
            "Llama input layernorm mapping mismatch");
    require(no_bias.at("layers.0.self_attn.o_proj.weight") ==
                "model.layers.0.self_attn.o_proj.weight",
            "Llama o_proj mapping mismatch");

    const auto tied_with_bias = ops::LlamaKeyMapper::generate_llama_mapping(1, true, true);
    require(tied_with_bias.count("lm_head.weight") == 0,
            "Llama tied mapper should skip lm_head.weight");
    require(tied_with_bias.count("layers.0.self_attn.q_proj.bias") == 1,
            "Llama attention_bias=true mapper should include q_proj.bias");
    require(tied_with_bias.count("layers.0.self_attn.o_proj.bias") == 1,
            "Llama attention_bias=true mapper should include o_proj.bias");
}

void test_llama_rejects_unsupported_rope_scaling() {
    const fs::path root = make_case_dir("llama_rope_scaling");
    write_file(root / "config.json",
               R"({"model_type":"llama","tie_word_embeddings":false,"vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rope_scaling":{"type":"linear","factor":2.0}})");

    bool threw = false;
    try {
        (void)ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("rope_scaling") != std::string::npos;
    }
    require(threw, "Llama unsupported rope_scaling type should be rejected");
}

void test_mistral_is_recognized_but_graph_is_not_claimed() {
    const fs::path root = make_case_dir("mistral");
    write_file(root / "config.json",
               R"({"model_type":"mistral","tie_word_embeddings":false,"vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4})");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Mistral, "Mistral registry inference failed");

    bool threw = false;
    try {
        (void)ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("Mistral") != std::string::npos &&
                std::string(e.what()).find("not passed yet") != std::string::npos;
    }
    require(threw, "Mistral AutoModel should fail explicitly until graph gates pass");
}

void test_qwen_untied_lm_head_rejected() {
    const fs::path root = make_case_dir("qwen_untied");
    write_file(root / "config.json",
               R"({"model_type":"qwen2","tie_word_embeddings":false,"vocab_size":32,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16})");

    bool threw = false;
    try {
        (void)ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("tie_word_embeddings=false") != std::string::npos;
    }
    require(threw, "Qwen untied lm_head checkpoint should be rejected");
}

void test_assign_weight_shape_check() {
    ops::GPT2Config cfg;
    cfg.vocab_size = 8;
    cfg.n_positions = 4;
    cfg.n_embd = 4;
    cfg.n_layer = 1;
    cfg.n_head = 1;
    ops::GPT2Model model(cfg);

    auto wrong_wte = std::make_shared<ops::Tensor>(
        std::vector<int64_t>{7, 4}, ops::kFloat32, ops::kCPU);

    bool threw = false;
    try {
        model.assign_weight("wte.weight", wrong_wte);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("shape mismatch") != std::string::npos;
    }
    require(threw, "GPT-2 strict assign_weight shape check did not throw");

    model.assign_weight("wte.weight", wrong_wte, false);
}

}  // namespace

int main() {
    test_gpt2_auto_trainer_step();
    test_gpt2_lora_target_policy();
    test_gemma_auto_lora_dispatch();
    test_qwen_auto_lora_dispatch();
    test_llama_auto_trainer_step_with_untied_head();
    test_llama_key_mapper_respects_bias_and_lm_head();
    test_llama_rejects_unsupported_rope_scaling();
    test_mistral_is_recognized_but_graph_is_not_claimed();
    test_qwen_untied_lm_head_rejected();
    test_assign_weight_shape_check();
    return 0;
}
