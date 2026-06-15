/**
 * @file dpo_loss.h
 * @brief Direct Preference Optimization loss for causal language models.
 */

#pragma once

#include "tensor.h"

#include <string>
#include <vector>

namespace ops {

struct DPOMetrics {
    float loss = 0.0f;
    float chosen_reward = 0.0f;
    float rejected_reward = 0.0f;
    float reward_margin = 0.0f;
    float reward_accuracy = 0.0f;
};

/**
 * @brief Sum response-token log probabilities from dense [B,S,V] logits.
 *
 * This follows the standard causal-LM shift:
 * logits[:, :-1, :] scores input_ids[:, 1:], and response_mask[:, 1:]
 * decides which response tokens contribute to the sequence log probability.
 */
std::vector<float> sequence_logprob_masked(const TensorPtr& logits,
                                           const TensorPtr& input_ids,
                                           const TensorPtr& response_mask);

/**
 * @brief Sum response-token log probabilities without materializing logits.
 *
 * hidden is [B,S,H], lm_head_weight is [V,H]. The result matches
 * sequence_logprob_masked(matmul_rhs_T(hidden, lm_head_weight), ...).
 */
std::vector<float> sequence_logprob_masked_from_hidden(const TensorPtr& hidden,
                                                       const TensorPtr& lm_head_weight,
                                                       const TensorPtr& input_ids,
                                                       const TensorPtr& response_mask);

/**
 * @brief DPO loss from dense policy/reference logits.
 *
 * Gradients flow only through policy_chosen_logits and policy_rejected_logits.
 * Reference logits are treated as frozen scores.
 */
TensorPtr dpo_loss(const TensorPtr& policy_chosen_logits,
                   const TensorPtr& policy_rejected_logits,
                   const TensorPtr& ref_chosen_logits,
                   const TensorPtr& ref_rejected_logits,
                   const TensorPtr& chosen_input_ids,
                   const TensorPtr& rejected_input_ids,
                   const TensorPtr& chosen_response_mask,
                   const TensorPtr& rejected_response_mask,
                   float beta);

/**
 * @brief DPO loss from dense policy logits and cached frozen-reference logps.
 */
TensorPtr dpo_loss_with_ref_logps(const TensorPtr& policy_chosen_logits,
                                  const TensorPtr& policy_rejected_logits,
                                  const std::vector<float>& ref_chosen_logps,
                                  const std::vector<float>& ref_rejected_logps,
                                  const TensorPtr& chosen_input_ids,
                                  const TensorPtr& rejected_input_ids,
                                  const TensorPtr& chosen_response_mask,
                                  const TensorPtr& rejected_response_mask,
                                  float beta);

/**
 * @brief Memory-first DPO loss from policy hidden states and cached ref logps.
 *
 * This avoids allocating [B,S,V] logits and is the preferred mobile path.
 * Gradients flow through chosen_hidden, rejected_hidden, and lm_head_weight
 * when those tensors require gradients.
 */
TensorPtr streaming_dpo_loss_with_ref_logps(const TensorPtr& chosen_hidden,
                                            const TensorPtr& rejected_hidden,
                                            const TensorPtr& lm_head_weight,
                                            const std::vector<float>& ref_chosen_logps,
                                            const std::vector<float>& ref_rejected_logps,
                                            const TensorPtr& chosen_input_ids,
                                            const TensorPtr& rejected_input_ids,
                                            const TensorPtr& chosen_response_mask,
                                            const TensorPtr& rejected_response_mask,
                                            float beta);

DPOMetrics compute_dpo_metrics_from_logps(const std::vector<float>& policy_chosen_logps,
                                          const std::vector<float>& policy_rejected_logps,
                                          const std::vector<float>& ref_chosen_logps,
                                          const std::vector<float>& ref_rejected_logps,
                                          float beta);

DPOMetrics compute_dpo_metrics_from_logits(const TensorPtr& policy_chosen_logits,
                                           const TensorPtr& policy_rejected_logits,
                                           const TensorPtr& ref_chosen_logits,
                                           const TensorPtr& ref_rejected_logits,
                                           const TensorPtr& chosen_input_ids,
                                           const TensorPtr& rejected_input_ids,
                                           const TensorPtr& chosen_response_mask,
                                           const TensorPtr& rejected_response_mask,
                                           float beta);

DPOMetrics compute_dpo_metrics_from_policy_logits_and_ref_logps(
    const TensorPtr& policy_chosen_logits,
    const TensorPtr& policy_rejected_logits,
    const std::vector<float>& ref_chosen_logps,
    const std::vector<float>& ref_rejected_logps,
    const TensorPtr& chosen_input_ids,
    const TensorPtr& rejected_input_ids,
    const TensorPtr& chosen_response_mask,
    const TensorPtr& rejected_response_mask,
    float beta);

DPOMetrics compute_dpo_metrics_from_policy_hidden_and_ref_logps(
    const TensorPtr& chosen_hidden,
    const TensorPtr& rejected_hidden,
    const TensorPtr& lm_head_weight,
    const std::vector<float>& ref_chosen_logps,
    const std::vector<float>& ref_rejected_logps,
    const TensorPtr& chosen_input_ids,
    const TensorPtr& rejected_input_ids,
    const TensorPtr& chosen_response_mask,
    const TensorPtr& rejected_response_mask,
    float beta);

}  // namespace ops
