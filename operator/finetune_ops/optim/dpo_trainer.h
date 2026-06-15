#pragma once

#include "../core/dpo_loss.h"
#include "../data/preference_batch.h"
#include "../graph/auto_model.h"
#include "adam.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace ops {

struct DPOTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    float beta = 0.1f;
    bool use_streaming_dpo_loss = true;
    int gradient_accumulation_steps = 1;
};

struct DPOTrainStepResult {
    float loss = 0.0f;
    float accumulated_loss = 0.0f;
    float chosen_reward = 0.0f;
    float rejected_reward = 0.0f;
    float reward_margin = 0.0f;
    float reward_accuracy = 0.0f;
    int trainable_tensor_count = 0;
    int pair_count = 0;
    int valid_response_token_count = 0;
    int accumulation_step = 0;
    int gradient_accumulation_steps = 1;
    bool optimizer_step = true;
};

struct DPOFitConfig {
    int max_steps = 1;
    std::size_t batch_size = 1;
    bool loop_dataset = true;
};

struct DPOFitStep {
    int step = 0;
    DPOTrainStepResult train_result;
};

struct DPOFitResult {
    int completed_steps = 0;
    bool stopped_early = false;
    float final_loss = 0.0f;
    float mean_loss = 0.0f;
    float final_reward_margin = 0.0f;
    float final_reward_accuracy = 0.0f;
    int trainable_tensor_count = 0;
    int total_pairs = 0;
};

using DPOFitStepCallback = std::function<void(const DPOFitStep&)>;

class DPOTrainer {
public:
    DPOTrainer(AutoModelForCausalLM& policy_model,
               const DPOTrainerConfig& config = DPOTrainerConfig());

    DPOTrainer(AutoModelForCausalLM& policy_model,
               AutoModelForCausalLM& reference_model,
               const DPOTrainerConfig& config = DPOTrainerConfig());

    DPOTrainStepResult train_step(const PreferenceBatch& batch);

    DPOFitResult fit(PreferenceBatchProvider& provider,
                     const DPOFitConfig& fit_config,
                     DPOFitStepCallback on_step = nullptr);

private:
    AutoModelForCausalLM& policy_model_;
    AutoModelForCausalLM* reference_model_ = nullptr;
    DPOTrainerConfig config_;
    std::unique_ptr<Adam> optimizer_;
    int accum_counter_ = 0;
    float accum_loss_ = 0.0f;
};

}  // namespace ops
