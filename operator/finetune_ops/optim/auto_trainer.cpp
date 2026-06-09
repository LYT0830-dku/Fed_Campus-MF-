#include "auto_trainer.h"

#include "../core/lm_loss.h"

#include <stdexcept>
#include <vector>

namespace ops {

namespace {

int count_valid_shifted_labels(const TensorPtr& labels, int ignore_index) {
    if (!labels || labels->ndim() != 2 || labels->dtype() != kInt32) {
        return 0;
    }
    const auto& shape = labels->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int32_t* data = labels->data<int32_t>();
    int valid = 0;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 1; s < S; ++s) {
            if (data[b * S + s] != ignore_index) {
                ++valid;
            }
        }
    }
    return valid;
}

}  // namespace

AutoTrainer::AutoTrainer(AutoModelForCausalLM& model,
                         const AutoTrainerConfig& config)
    : model_(model), config_(config) {
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

AutoTrainStepResult AutoTrainer::train_step(const TensorPtr& input_ids,
                                            const TensorPtr& attention_mask,
                                            const TensorPtr& labels) {
    auto params = model_.trainable_parameters();
    if (params.empty()) {
        throw std::runtime_error("AutoTrainer::train_step requires trainable parameters; call init_lora first");
    }
    const int valid_label_count = count_valid_shifted_labels(labels, config_.ignore_index);
    if (valid_label_count <= 0) {
        throw std::invalid_argument("AutoTrainer::train_step requires at least one valid shifted label");
    }

    TensorPtr loss;
    if (config_.use_streaming_lm_loss) {
        auto hidden = model_.forward_hidden(input_ids, attention_mask);
        auto weight = model_.lm_head_weight_for_loss();
        loss = streaming_lm_cross_entropy(hidden, weight, labels, config_.ignore_index, "mean");
    } else {
        auto logits = model_.forward(input_ids, attention_mask);
        loss = lm_cross_entropy(logits, labels, config_.ignore_index, "mean");
    }
    loss->backward();

    optimizer_->clip_grad_norm(params, config_.max_grad_norm);

    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(params, grads);
    optimizer_->zero_grad(params);

    AutoTrainStepResult result;
    result.loss = loss->data<float>()[0];
    result.trainable_tensor_count = static_cast<int>(params.size());
    result.valid_label_count = valid_label_count;
    return result;
}

AutoTrainStepResult AutoTrainer::train_step(const CausalLMBatch& batch) {
    if (!batch.input_ids || !batch.attention_mask || !batch.labels) {
        throw std::invalid_argument("AutoTrainer::train_step received an incomplete CausalLMBatch");
    }
    return train_step(batch.input_ids, batch.attention_mask, batch.labels);
}

}  // namespace ops
