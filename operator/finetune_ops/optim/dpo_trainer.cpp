#include "dpo_trainer.h"

#include "../core/ops.h"

#include <stdexcept>
#include <vector>

namespace ops {

namespace {

void validate_config(const DPOTrainerConfig& config) {
    if (!(config.beta > 0.0f)) {
        throw std::invalid_argument("DPOTrainer requires beta > 0");
    }
    if (config.gradient_accumulation_steps <= 0) {
        throw std::invalid_argument("DPOTrainer requires gradient_accumulation_steps > 0");
    }
}

void validate_batch(const PreferenceBatch& batch) {
    if (!batch.chosen_input_ids || !batch.chosen_attention_mask || !batch.chosen_response_mask ||
        !batch.rejected_input_ids || !batch.rejected_attention_mask || !batch.rejected_response_mask) {
        throw std::invalid_argument("DPOTrainer::train_step received an incomplete PreferenceBatch");
    }
    if (batch.batch_size <= 0) {
        throw std::invalid_argument("DPOTrainer::train_step requires at least one preference pair");
    }
}

}  // namespace

DPOTrainer::DPOTrainer(AutoModelForCausalLM& policy_model,
                       const DPOTrainerConfig& config)
    : policy_model_(policy_model), config_(config) {
    validate_config(config_);
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

DPOTrainer::DPOTrainer(AutoModelForCausalLM& policy_model,
                       AutoModelForCausalLM& reference_model,
                       const DPOTrainerConfig& config)
    : policy_model_(policy_model), reference_model_(&reference_model), config_(config) {
    validate_config(config_);
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

DPOTrainStepResult DPOTrainer::train_step(const PreferenceBatch& batch) {
    validate_batch(batch);
    auto params = policy_model_.trainable_parameters();
    if (params.empty()) {
        throw std::runtime_error("DPOTrainer::train_step requires trainable parameters; call init_lora first");
    }

    TensorPtr loss;
    DPOMetrics metrics;

    if (config_.use_streaming_dpo_loss) {
        auto chosen_hidden =
            policy_model_.forward_hidden(batch.chosen_input_ids, batch.chosen_attention_mask);
        auto rejected_hidden =
            policy_model_.forward_hidden(batch.rejected_input_ids, batch.rejected_attention_mask);
        auto lm_head_weight = policy_model_.lm_head_weight_for_loss();

        std::vector<float> ref_chosen_logps;
        std::vector<float> ref_rejected_logps;
        if (batch.has_cached_reference_logps()) {
            ref_chosen_logps = batch.ref_chosen_logps;
            ref_rejected_logps = batch.ref_rejected_logps;
        } else if (reference_model_) {
            auto ref_chosen_hidden =
                reference_model_->forward_hidden(batch.chosen_input_ids, batch.chosen_attention_mask);
            auto ref_rejected_hidden =
                reference_model_->forward_hidden(batch.rejected_input_ids, batch.rejected_attention_mask);
            auto ref_weight = reference_model_->lm_head_weight_for_loss();
            ref_chosen_logps = sequence_logprob_masked_from_hidden(
                ref_chosen_hidden, ref_weight, batch.chosen_input_ids, batch.chosen_response_mask);
            ref_rejected_logps = sequence_logprob_masked_from_hidden(
                ref_rejected_hidden, ref_weight, batch.rejected_input_ids, batch.rejected_response_mask);
        } else {
            throw std::invalid_argument(
                "DPOTrainer::train_step requires cached reference logps or a reference model");
        }

        loss = streaming_dpo_loss_with_ref_logps(
            chosen_hidden,
            rejected_hidden,
            lm_head_weight,
            ref_chosen_logps,
            ref_rejected_logps,
            batch.chosen_input_ids,
            batch.rejected_input_ids,
            batch.chosen_response_mask,
            batch.rejected_response_mask,
            config_.beta);
        metrics = compute_dpo_metrics_from_policy_hidden_and_ref_logps(
            chosen_hidden,
            rejected_hidden,
            lm_head_weight,
            ref_chosen_logps,
            ref_rejected_logps,
            batch.chosen_input_ids,
            batch.rejected_input_ids,
            batch.chosen_response_mask,
            batch.rejected_response_mask,
            config_.beta);
    } else {
        auto policy_chosen_logits =
            policy_model_.forward(batch.chosen_input_ids, batch.chosen_attention_mask);
        auto policy_rejected_logits =
            policy_model_.forward(batch.rejected_input_ids, batch.rejected_attention_mask);

        if (batch.has_cached_reference_logps()) {
            loss = dpo_loss_with_ref_logps(
                policy_chosen_logits,
                policy_rejected_logits,
                batch.ref_chosen_logps,
                batch.ref_rejected_logps,
                batch.chosen_input_ids,
                batch.rejected_input_ids,
                batch.chosen_response_mask,
                batch.rejected_response_mask,
                config_.beta);
            metrics = compute_dpo_metrics_from_policy_logits_and_ref_logps(
                policy_chosen_logits,
                policy_rejected_logits,
                batch.ref_chosen_logps,
                batch.ref_rejected_logps,
                batch.chosen_input_ids,
                batch.rejected_input_ids,
                batch.chosen_response_mask,
                batch.rejected_response_mask,
                config_.beta);
        } else if (reference_model_) {
            auto ref_chosen_logits =
                reference_model_->forward(batch.chosen_input_ids, batch.chosen_attention_mask);
            auto ref_rejected_logits =
                reference_model_->forward(batch.rejected_input_ids, batch.rejected_attention_mask);
            loss = dpo_loss(
                policy_chosen_logits,
                policy_rejected_logits,
                ref_chosen_logits,
                ref_rejected_logits,
                batch.chosen_input_ids,
                batch.rejected_input_ids,
                batch.chosen_response_mask,
                batch.rejected_response_mask,
                config_.beta);
            metrics = compute_dpo_metrics_from_logits(
                policy_chosen_logits,
                policy_rejected_logits,
                ref_chosen_logits,
                ref_rejected_logits,
                batch.chosen_input_ids,
                batch.rejected_input_ids,
                batch.chosen_response_mask,
                batch.rejected_response_mask,
                config_.beta);
        } else {
            throw std::invalid_argument(
                "DPOTrainer::train_step requires cached reference logps or a reference model");
        }
    }

    const float raw_loss = loss->data<float>()[0];
    TensorPtr backward_loss = loss;
    if (config_.gradient_accumulation_steps > 1) {
        backward_loss = mul(loss, 1.0f / static_cast<float>(config_.gradient_accumulation_steps));
    }
    backward_loss->backward();

    accum_counter_ += 1;
    accum_loss_ += raw_loss;

    DPOTrainStepResult result;
    result.loss = raw_loss;
    result.chosen_reward = metrics.chosen_reward;
    result.rejected_reward = metrics.rejected_reward;
    result.reward_margin = metrics.reward_margin;
    result.reward_accuracy = metrics.reward_accuracy;
    result.trainable_tensor_count = static_cast<int>(params.size());
    result.pair_count = batch.batch_size;
    result.valid_response_token_count = batch.valid_response_token_count;
    result.accumulation_step = accum_counter_;
    result.gradient_accumulation_steps = config_.gradient_accumulation_steps;
    result.optimizer_step = accum_counter_ >= config_.gradient_accumulation_steps;

    if (!result.optimizer_step) {
        return result;
    }

    optimizer_->clip_grad_norm(params, config_.max_grad_norm);
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(params, grads);
    optimizer_->zero_grad(params);

    result.accumulated_loss = accum_loss_ / static_cast<float>(config_.gradient_accumulation_steps);
    accum_counter_ = 0;
    accum_loss_ = 0.0f;
    return result;
}

DPOFitResult DPOTrainer::fit(PreferenceBatchProvider& provider,
                             const DPOFitConfig& fit_config,
                             DPOFitStepCallback on_step) {
    if (fit_config.max_steps <= 0) {
        throw std::invalid_argument("DPOTrainer::fit requires max_steps > 0");
    }
    if (fit_config.batch_size == 0) {
        throw std::invalid_argument("DPOTrainer::fit requires batch_size > 0");
    }

    DPOFitResult result;
    float loss_sum = 0.0f;
    for (int step = 1; step <= fit_config.max_steps; ++step) {
        PreferenceBatch batch = provider.next_batch(fit_config.batch_size, fit_config.loop_dataset);
        if (!batch.chosen_input_ids) {
            result.stopped_early = true;
            break;
        }

        DPOTrainStepResult step_result = train_step(batch);
        result.completed_steps += 1;
        result.final_loss = step_result.loss;
        result.final_reward_margin = step_result.reward_margin;
        result.final_reward_accuracy = step_result.reward_accuracy;
        result.trainable_tensor_count = step_result.trainable_tensor_count;
        result.total_pairs += step_result.pair_count;
        loss_sum += step_result.loss;

        if (on_step) {
            on_step(DPOFitStep{result.completed_steps, step_result});
        }
    }

    if (result.completed_steps > 0) {
        result.mean_loss = loss_sum / static_cast<float>(result.completed_steps);
    }
    return result;
}

}  // namespace ops
