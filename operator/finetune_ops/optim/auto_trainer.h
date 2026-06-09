#pragma once

#include "../core/tensor.h"
#include "../data/batch_provider.h"
#include "../data/causal_lm_batch.h"
#include "../graph/auto_model.h"
#include "adam.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace ops {

struct AutoTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    int ignore_index = -100;
    bool use_streaming_lm_loss = true;
    int gradient_accumulation_steps = 1;
};

struct AutoTrainStepResult {
    float loss = 0.0f;
    float accumulated_loss = 0.0f;
    int trainable_tensor_count = 0;
    int valid_label_count = 0;
    int accumulation_step = 0;
    int gradient_accumulation_steps = 1;
    bool optimizer_step = true;
};

struct AutoFitConfig {
    int max_steps = 1;
    std::size_t batch_size = 1;
    bool loop_dataset = true;
};

struct AutoFitStep {
    int step = 0;
    AutoTrainStepResult train_result;
};

struct AutoFitResult {
    int completed_steps = 0;
    bool stopped_early = false;
    float final_loss = 0.0f;
    float mean_loss = 0.0f;
    int trainable_tensor_count = 0;
    int total_valid_label_count = 0;
};

using AutoFitStepCallback = std::function<void(const AutoFitStep&)>;

class AutoTrainer {
public:
    AutoTrainer(AutoModelForCausalLM& model,
                const AutoTrainerConfig& config = AutoTrainerConfig());

    AutoTrainStepResult train_step(const TensorPtr& input_ids,
                                   const TensorPtr& attention_mask,
                                   const TensorPtr& labels);

    AutoTrainStepResult train_step(const CausalLMBatch& batch);

    AutoFitResult fit(BatchProvider& provider,
                      const AutoFitConfig& fit_config,
                      AutoFitStepCallback on_step = nullptr);

private:
    AutoModelForCausalLM& model_;
    AutoTrainerConfig config_;
    std::unique_ptr<Adam> optimizer_;
    int accum_counter_ = 0;
    float accum_loss_ = 0.0f;
};

}  // namespace ops
