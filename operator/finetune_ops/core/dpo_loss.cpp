/**
 * @file dpo_loss.cpp
 * @brief Direct Preference Optimization loss implementation.
 */

#include "dpo_loss.h"

#include "backward_functions.h"
#include "ops.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ops {

namespace {

double neg_log_sigmoid(double x) {
    if (x >= 0.0) {
        return std::log1p(std::exp(-x));
    }
    return -x + std::log1p(std::exp(x));
}

double sigmoid_minus_one(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return -z / (1.0 + z);
    }
    const double z = std::exp(x);
    return -1.0 / (1.0 + z);
}

float row_value(const TensorPtr& tensor, int64_t offset) {
    if (tensor->dtype() == kFloat32) {
        return tensor->data<float>()[offset];
    }
    if (tensor->dtype() == kFloat16) {
        return fp16_bits_to_float32(tensor->data<uint16_t>()[offset]);
    }
    if (tensor->dtype() == kBFloat16) {
        return bf16_bits_to_float32(tensor->data<uint16_t>()[offset]);
    }
    throw std::runtime_error("dpo_loss: unsupported floating tensor dtype");
}

void validate_logits_inputs(const TensorPtr& logits,
                            const TensorPtr& input_ids,
                            const TensorPtr& response_mask,
                            const char* name) {
    if (!logits || !input_ids || !response_mask) {
        throw std::runtime_error(std::string(name) + ": null input");
    }
    if (logits->dtype() != kFloat32) {
        throw std::runtime_error(std::string(name) + ": logits must be float32");
    }
    if (logits->ndim() != 3) {
        throw std::runtime_error(std::string(name) + ": logits must be [B,S,V]");
    }
    if (input_ids->ndim() != 2 || response_mask->ndim() != 2) {
        throw std::runtime_error(std::string(name) + ": input_ids and response_mask must be [B,S]");
    }
    if (input_ids->dtype() != kInt32 || response_mask->dtype() != kInt32) {
        throw std::runtime_error(std::string(name) + ": input_ids and response_mask must be int32");
    }

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    if (input_ids->shape()[0] != B || input_ids->shape()[1] != S ||
        response_mask->shape()[0] != B || response_mask->shape()[1] != S) {
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    }
}

void validate_hidden_inputs(const TensorPtr& hidden,
                            const TensorPtr& lm_head_weight,
                            const TensorPtr& input_ids,
                            const TensorPtr& response_mask,
                            const char* name) {
    if (!hidden || !lm_head_weight || !input_ids || !response_mask) {
        throw std::runtime_error(std::string(name) + ": null input");
    }
    if (hidden->dtype() != kFloat32) {
        throw std::runtime_error(std::string(name) + ": hidden must be float32");
    }
    if (hidden->ndim() != 3) {
        throw std::runtime_error(std::string(name) + ": hidden must be [B,S,H]");
    }
    if (lm_head_weight->ndim() != 2) {
        throw std::runtime_error(std::string(name) + ": lm_head_weight must be [V,H]");
    }
    if (input_ids->ndim() != 2 || response_mask->ndim() != 2) {
        throw std::runtime_error(std::string(name) + ": input_ids and response_mask must be [B,S]");
    }
    if (input_ids->dtype() != kInt32 || response_mask->dtype() != kInt32) {
        throw std::runtime_error(std::string(name) + ": input_ids and response_mask must be int32");
    }

    const auto& hshape = hidden->shape();
    const auto& wshape = lm_head_weight->shape();
    const int64_t B = hshape[0];
    const int64_t S = hshape[1];
    const int64_t H = hshape[2];
    if (wshape[1] != H) {
        throw std::runtime_error(std::string(name) + ": hidden H and weight H mismatch");
    }
    if (input_ids->shape()[0] != B || input_ids->shape()[1] != S ||
        response_mask->shape()[0] != B || response_mask->shape()[1] != S) {
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    }
}

std::vector<float> compute_sequence_logps_logits(const TensorPtr& logits,
                                                 const TensorPtr& input_ids,
                                                 const TensorPtr& response_mask) {
    validate_logits_inputs(logits, input_ids, response_mask, "sequence_logprob_masked");

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];

    const float* x = logits->data<float>();
    const int32_t* ids = input_ids->data<int32_t>();
    const int32_t* mask = response_mask->data<int32_t>();

    std::vector<float> logps(static_cast<size_t>(B), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        double sum = 0.0;
        for (int64_t s = 0; s + 1 < S; ++s) {
            if (mask[b * S + (s + 1)] == 0) {
                continue;
            }
            const int32_t cls = ids[b * S + (s + 1)];
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("sequence_logprob_masked: token id out of range");
            }

            const float* row = x + (b * S + s) * V;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                max_val = std::max(max_val, row[v]);
            }
            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(static_cast<double>(row[v] - max_val));
            }
            sum += static_cast<double>(row[cls]) -
                   (static_cast<double>(max_val) + std::log(denom));
        }
        logps[static_cast<size_t>(b)] = static_cast<float>(sum);
    }
    return logps;
}

std::vector<float> compute_sequence_logps_hidden(const TensorPtr& hidden,
                                                 const TensorPtr& lm_head_weight,
                                                 const TensorPtr& input_ids,
                                                 const TensorPtr& response_mask) {
    validate_hidden_inputs(hidden, lm_head_weight, input_ids, response_mask,
                           "sequence_logprob_masked_from_hidden");

    const auto& hshape = hidden->shape();
    const auto& wshape = lm_head_weight->shape();
    const int64_t B = hshape[0];
    const int64_t S = hshape[1];
    const int64_t H = hshape[2];
    const int64_t V = wshape[0];

    const float* hidden_data = hidden->data<float>();
    const int32_t* ids = input_ids->data<int32_t>();
    const int32_t* mask = response_mask->data<int32_t>();
    std::vector<float> logits(static_cast<size_t>(V), 0.0f);
    std::vector<float> logps(static_cast<size_t>(B), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        double sum = 0.0;
        for (int64_t s = 0; s + 1 < S; ++s) {
            if (mask[b * S + (s + 1)] == 0) {
                continue;
            }
            const int32_t cls = ids[b * S + (s + 1)];
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("sequence_logprob_masked_from_hidden: token id out of range");
            }

            const float* hrow = hidden_data + (b * S + s) * H;
            float max_val = -std::numeric_limits<float>::infinity();
            float target = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                double z = 0.0;
                const int64_t wbase = v * H;
                for (int64_t h = 0; h < H; ++h) {
                    z += static_cast<double>(hrow[h]) *
                         static_cast<double>(row_value(lm_head_weight, wbase + h));
                }
                logits[static_cast<size_t>(v)] = static_cast<float>(z);
                if (v == cls) {
                    target = static_cast<float>(z);
                }
                max_val = std::max(max_val, static_cast<float>(z));
            }

            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(static_cast<double>(logits[static_cast<size_t>(v)] - max_val));
            }
            sum += static_cast<double>(target) -
                   (static_cast<double>(max_val) + std::log(denom));
        }
        logps[static_cast<size_t>(b)] = static_cast<float>(sum);
    }
    return logps;
}

void validate_dpo_logps(const std::vector<float>& policy_chosen_logps,
                        const std::vector<float>& policy_rejected_logps,
                        const std::vector<float>& ref_chosen_logps,
                        const std::vector<float>& ref_rejected_logps,
                        const char* name) {
    const size_t n = policy_chosen_logps.size();
    if (n == 0 || policy_rejected_logps.size() != n ||
        ref_chosen_logps.size() != n || ref_rejected_logps.size() != n) {
        throw std::runtime_error(std::string(name) + ": inconsistent batch sizes");
    }
}

std::pair<float, std::pair<std::vector<double>, std::vector<double>>> compute_loss_and_coeffs(
    const std::vector<float>& policy_chosen_logps,
    const std::vector<float>& policy_rejected_logps,
    const std::vector<float>& ref_chosen_logps,
    const std::vector<float>& ref_rejected_logps,
    float beta,
    const char* name) {
    validate_dpo_logps(policy_chosen_logps, policy_rejected_logps,
                       ref_chosen_logps, ref_rejected_logps, name);
    if (!(beta > 0.0f)) {
        throw std::runtime_error(std::string(name) + ": beta must be positive");
    }

    const size_t n = policy_chosen_logps.size();
    const double inv_n = 1.0 / static_cast<double>(n);
    double loss_sum = 0.0;
    std::vector<double> chosen_coeffs(n, 0.0);
    std::vector<double> rejected_coeffs(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        const double pi_logratio =
            static_cast<double>(policy_chosen_logps[i]) -
            static_cast<double>(policy_rejected_logps[i]);
        const double ref_logratio =
            static_cast<double>(ref_chosen_logps[i]) -
            static_cast<double>(ref_rejected_logps[i]);
        const double margin = static_cast<double>(beta) * (pi_logratio - ref_logratio);

        loss_sum += neg_log_sigmoid(margin);

        const double dloss_dmargin = sigmoid_minus_one(margin);
        const double dloss_dseq = dloss_dmargin * static_cast<double>(beta) * inv_n;
        chosen_coeffs[i] = dloss_dseq;
        rejected_coeffs[i] = -dloss_dseq;
    }

    return {
        static_cast<float>(loss_sum * inv_n),
        {std::move(chosen_coeffs), std::move(rejected_coeffs)},
    };
}

void accumulate_logits_logp_grads(const TensorPtr& grad_logits,
                                  const TensorPtr& logits,
                                  const TensorPtr& input_ids,
                                  const TensorPtr& response_mask,
                                  const std::vector<double>& sample_coeffs) {
    validate_logits_inputs(logits, input_ids, response_mask, "accumulate_logits_logp_grads");

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];
    if (static_cast<int64_t>(sample_coeffs.size()) != B) {
        throw std::runtime_error("accumulate_logits_logp_grads: coeff size mismatch");
    }

    const float* x = logits->data<float>();
    const int32_t* ids = input_ids->data<int32_t>();
    const int32_t* mask = response_mask->data<int32_t>();
    float* grad = grad_logits->data<float>();

    for (int64_t b = 0; b < B; ++b) {
        const double coeff = sample_coeffs[static_cast<size_t>(b)];
        if (coeff == 0.0) {
            continue;
        }
        for (int64_t s = 0; s + 1 < S; ++s) {
            if (mask[b * S + (s + 1)] == 0) {
                continue;
            }
            const int32_t cls = ids[b * S + (s + 1)];
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("accumulate_logits_logp_grads: token id out of range");
            }

            const float* row = x + (b * S + s) * V;
            float* grow = grad + (b * S + s) * V;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                max_val = std::max(max_val, row[v]);
            }
            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(static_cast<double>(row[v] - max_val));
            }
            const double inv_denom = 1.0 / denom;
            for (int64_t v = 0; v < V; ++v) {
                const double p = std::exp(static_cast<double>(row[v] - max_val)) * inv_denom;
                const double one_hot = (v == cls) ? 1.0 : 0.0;
                grow[v] += static_cast<float>(coeff * (one_hot - p));
            }
        }
    }
}

void accumulate_hidden_logp_grads(const TensorPtr& grad_hidden,
                                  const TensorPtr& grad_weight,
                                  const TensorPtr& hidden,
                                  const TensorPtr& lm_head_weight,
                                  const TensorPtr& input_ids,
                                  const TensorPtr& response_mask,
                                  const std::vector<double>& sample_coeffs) {
    validate_hidden_inputs(hidden, lm_head_weight, input_ids, response_mask,
                           "accumulate_hidden_logp_grads");

    const auto& hshape = hidden->shape();
    const auto& wshape = lm_head_weight->shape();
    const int64_t B = hshape[0];
    const int64_t S = hshape[1];
    const int64_t H = hshape[2];
    const int64_t V = wshape[0];
    if (static_cast<int64_t>(sample_coeffs.size()) != B) {
        throw std::runtime_error("accumulate_hidden_logp_grads: coeff size mismatch");
    }

    const float* hidden_data = hidden->data<float>();
    const int32_t* ids = input_ids->data<int32_t>();
    const int32_t* mask = response_mask->data<int32_t>();
    float* grad_hidden_data = grad_hidden ? grad_hidden->data<float>() : nullptr;
    float* grad_weight_data = grad_weight ? grad_weight->data<float>() : nullptr;
    std::vector<float> logits(static_cast<size_t>(V), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        const double coeff = sample_coeffs[static_cast<size_t>(b)];
        if (coeff == 0.0) {
            continue;
        }
        for (int64_t s = 0; s + 1 < S; ++s) {
            if (mask[b * S + (s + 1)] == 0) {
                continue;
            }
            const int32_t cls = ids[b * S + (s + 1)];
            if (cls < 0 || cls >= V) {
                throw std::runtime_error("accumulate_hidden_logp_grads: token id out of range");
            }

            const float* hrow = hidden_data + (b * S + s) * H;
            float* ghrow = grad_hidden_data ? grad_hidden_data + (b * S + s) * H : nullptr;

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                double z = 0.0;
                const int64_t wbase = v * H;
                for (int64_t h = 0; h < H; ++h) {
                    z += static_cast<double>(hrow[h]) *
                         static_cast<double>(row_value(lm_head_weight, wbase + h));
                }
                logits[static_cast<size_t>(v)] = static_cast<float>(z);
                max_val = std::max(max_val, static_cast<float>(z));
            }

            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(static_cast<double>(logits[static_cast<size_t>(v)] - max_val));
            }
            const double inv_denom = 1.0 / denom;

            for (int64_t v = 0; v < V; ++v) {
                const double p = std::exp(static_cast<double>(logits[static_cast<size_t>(v)] - max_val)) * inv_denom;
                const double one_hot = (v == cls) ? 1.0 : 0.0;
                const double g = coeff * (one_hot - p);
                const int64_t wbase = v * H;

                if (ghrow) {
                    for (int64_t h = 0; h < H; ++h) {
                        ghrow[h] += static_cast<float>(
                            g * static_cast<double>(row_value(lm_head_weight, wbase + h)));
                    }
                }
                if (grad_weight_data) {
                    float* gwrow = grad_weight_data + wbase;
                    for (int64_t h = 0; h < H; ++h) {
                        gwrow[h] += static_cast<float>(g * static_cast<double>(hrow[h]));
                    }
                }
            }
        }
    }
}

class DPOLogitsBackward final : public BackwardFunction {
public:
    DPOLogitsBackward(TensorPtr policy_chosen_logits,
                      TensorPtr policy_rejected_logits,
                      TensorPtr chosen_input_ids,
                      TensorPtr rejected_input_ids,
                      TensorPtr chosen_response_mask,
                      TensorPtr rejected_response_mask,
                      std::vector<double> chosen_coeffs,
                      std::vector<double> rejected_coeffs)
        : policy_chosen_logits_(std::move(policy_chosen_logits)),
          policy_rejected_logits_(std::move(policy_rejected_logits)),
          chosen_input_ids_(std::move(chosen_input_ids)),
          rejected_input_ids_(std::move(rejected_input_ids)),
          chosen_response_mask_(std::move(chosen_response_mask)),
          rejected_response_mask_(std::move(rejected_response_mask)),
          chosen_coeffs_(std::move(chosen_coeffs)),
          rejected_coeffs_(std::move(rejected_coeffs)) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        float outer = 1.0f;
        if (grad_output && grad_output->numel() > 0) {
            outer = grad_output->data<float>()[0];
        }

        std::vector<double> chosen_coeffs = chosen_coeffs_;
        std::vector<double> rejected_coeffs = rejected_coeffs_;
        for (double& value : chosen_coeffs) value *= static_cast<double>(outer);
        for (double& value : rejected_coeffs) value *= static_cast<double>(outer);

        TensorPtr grad_chosen = policy_chosen_logits_->requires_grad()
            ? zeros(policy_chosen_logits_->shape(), kFloat32, kCPU)
            : nullptr;
        TensorPtr grad_rejected = policy_rejected_logits_->requires_grad()
            ? zeros(policy_rejected_logits_->shape(), kFloat32, kCPU)
            : nullptr;

        if (grad_chosen) {
            accumulate_logits_logp_grads(grad_chosen, policy_chosen_logits_,
                                         chosen_input_ids_, chosen_response_mask_,
                                         chosen_coeffs);
        }
        if (grad_rejected) {
            accumulate_logits_logp_grads(grad_rejected, policy_rejected_logits_,
                                         rejected_input_ids_, rejected_response_mask_,
                                         rejected_coeffs);
        }
        return {grad_chosen, grad_rejected};
    }

private:
    TensorPtr policy_chosen_logits_;
    TensorPtr policy_rejected_logits_;
    TensorPtr chosen_input_ids_;
    TensorPtr rejected_input_ids_;
    TensorPtr chosen_response_mask_;
    TensorPtr rejected_response_mask_;
    std::vector<double> chosen_coeffs_;
    std::vector<double> rejected_coeffs_;
};

class StreamingDPOBackward final : public BackwardFunction {
public:
    StreamingDPOBackward(TensorPtr chosen_hidden,
                         TensorPtr rejected_hidden,
                         TensorPtr lm_head_weight,
                         TensorPtr chosen_input_ids,
                         TensorPtr rejected_input_ids,
                         TensorPtr chosen_response_mask,
                         TensorPtr rejected_response_mask,
                         std::vector<double> chosen_coeffs,
                         std::vector<double> rejected_coeffs)
        : chosen_hidden_(std::move(chosen_hidden)),
          rejected_hidden_(std::move(rejected_hidden)),
          lm_head_weight_(std::move(lm_head_weight)),
          chosen_input_ids_(std::move(chosen_input_ids)),
          rejected_input_ids_(std::move(rejected_input_ids)),
          chosen_response_mask_(std::move(chosen_response_mask)),
          rejected_response_mask_(std::move(rejected_response_mask)),
          chosen_coeffs_(std::move(chosen_coeffs)),
          rejected_coeffs_(std::move(rejected_coeffs)) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        float outer = 1.0f;
        if (grad_output && grad_output->numel() > 0) {
            outer = grad_output->data<float>()[0];
        }

        std::vector<double> chosen_coeffs = chosen_coeffs_;
        std::vector<double> rejected_coeffs = rejected_coeffs_;
        for (double& value : chosen_coeffs) value *= static_cast<double>(outer);
        for (double& value : rejected_coeffs) value *= static_cast<double>(outer);

        TensorPtr grad_chosen_hidden = chosen_hidden_->requires_grad()
            ? zeros(chosen_hidden_->shape(), kFloat32, kCPU)
            : nullptr;
        TensorPtr grad_rejected_hidden = rejected_hidden_->requires_grad()
            ? zeros(rejected_hidden_->shape(), kFloat32, kCPU)
            : nullptr;
        TensorPtr grad_weight = lm_head_weight_->requires_grad()
            ? zeros(lm_head_weight_->shape(), kFloat32, kCPU)
            : nullptr;

        if (grad_chosen_hidden || grad_weight) {
            accumulate_hidden_logp_grads(grad_chosen_hidden, grad_weight,
                                         chosen_hidden_, lm_head_weight_,
                                         chosen_input_ids_, chosen_response_mask_,
                                         chosen_coeffs);
        }
        if (grad_rejected_hidden || grad_weight) {
            accumulate_hidden_logp_grads(grad_rejected_hidden, grad_weight,
                                         rejected_hidden_, lm_head_weight_,
                                         rejected_input_ids_, rejected_response_mask_,
                                         rejected_coeffs);
        }
        return {grad_chosen_hidden, grad_rejected_hidden, grad_weight, nullptr, nullptr, nullptr, nullptr};
    }

private:
    TensorPtr chosen_hidden_;
    TensorPtr rejected_hidden_;
    TensorPtr lm_head_weight_;
    TensorPtr chosen_input_ids_;
    TensorPtr rejected_input_ids_;
    TensorPtr chosen_response_mask_;
    TensorPtr rejected_response_mask_;
    std::vector<double> chosen_coeffs_;
    std::vector<double> rejected_coeffs_;
};

TensorPtr make_dense_dpo_loss_from_logps(const TensorPtr& policy_chosen_logits,
                                         const TensorPtr& policy_rejected_logits,
                                         const std::vector<float>& policy_chosen_logps,
                                         const std::vector<float>& policy_rejected_logps,
                                         const std::vector<float>& ref_chosen_logps,
                                         const std::vector<float>& ref_rejected_logps,
                                         const TensorPtr& chosen_input_ids,
                                         const TensorPtr& rejected_input_ids,
                                         const TensorPtr& chosen_response_mask,
                                         const TensorPtr& rejected_response_mask,
                                         float beta,
                                         const char* name) {
    auto computed = compute_loss_and_coeffs(policy_chosen_logps, policy_rejected_logps,
                                            ref_chosen_logps, ref_rejected_logps,
                                            beta, name);
    auto out = full({1}, computed.first, kFloat32, kCPU);
    if (policy_chosen_logits->requires_grad() || policy_rejected_logits->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<DPOLogitsBackward>(
            policy_chosen_logits, policy_rejected_logits,
            chosen_input_ids, rejected_input_ids,
            chosen_response_mask, rejected_response_mask,
            std::move(computed.second.first), std::move(computed.second.second));
#ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(out, {policy_chosen_logits, policy_rejected_logits}, backward_fn);
#else
        out->set_grad_fn([backward_fn, policy_chosen_logits, policy_rejected_logits](const TensorPtr& grad_output)
                             -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (policy_chosen_logits->requires_grad() && grads[0]) accumulate_gradient(policy_chosen_logits, grads[0]);
            if (policy_rejected_logits->requires_grad() && grads[1]) accumulate_gradient(policy_rejected_logits, grads[1]);
            return grads;
        });
#endif
    }
    return out;
}

}  // namespace

std::vector<float> sequence_logprob_masked(const TensorPtr& logits,
                                           const TensorPtr& input_ids,
                                           const TensorPtr& response_mask) {
    return compute_sequence_logps_logits(logits, input_ids, response_mask);
}

std::vector<float> sequence_logprob_masked_from_hidden(const TensorPtr& hidden,
                                                       const TensorPtr& lm_head_weight,
                                                       const TensorPtr& input_ids,
                                                       const TensorPtr& response_mask) {
    return compute_sequence_logps_hidden(hidden, lm_head_weight, input_ids, response_mask);
}

TensorPtr dpo_loss(const TensorPtr& policy_chosen_logits,
                   const TensorPtr& policy_rejected_logits,
                   const TensorPtr& ref_chosen_logits,
                   const TensorPtr& ref_rejected_logits,
                   const TensorPtr& chosen_input_ids,
                   const TensorPtr& rejected_input_ids,
                   const TensorPtr& chosen_response_mask,
                   const TensorPtr& rejected_response_mask,
                   float beta) {
    auto policy_chosen_logps =
        sequence_logprob_masked(policy_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto policy_rejected_logps =
        sequence_logprob_masked(policy_rejected_logits, rejected_input_ids, rejected_response_mask);
    auto ref_chosen_logps =
        sequence_logprob_masked(ref_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto ref_rejected_logps =
        sequence_logprob_masked(ref_rejected_logits, rejected_input_ids, rejected_response_mask);

    return make_dense_dpo_loss_from_logps(
        policy_chosen_logits, policy_rejected_logits,
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        chosen_input_ids, rejected_input_ids,
        chosen_response_mask, rejected_response_mask,
        beta, "dpo_loss");
}

TensorPtr dpo_loss_with_ref_logps(const TensorPtr& policy_chosen_logits,
                                  const TensorPtr& policy_rejected_logits,
                                  const std::vector<float>& ref_chosen_logps,
                                  const std::vector<float>& ref_rejected_logps,
                                  const TensorPtr& chosen_input_ids,
                                  const TensorPtr& rejected_input_ids,
                                  const TensorPtr& chosen_response_mask,
                                  const TensorPtr& rejected_response_mask,
                                  float beta) {
    auto policy_chosen_logps =
        sequence_logprob_masked(policy_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto policy_rejected_logps =
        sequence_logprob_masked(policy_rejected_logits, rejected_input_ids, rejected_response_mask);

    return make_dense_dpo_loss_from_logps(
        policy_chosen_logits, policy_rejected_logits,
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        chosen_input_ids, rejected_input_ids,
        chosen_response_mask, rejected_response_mask,
        beta, "dpo_loss_with_ref_logps");
}

TensorPtr streaming_dpo_loss_with_ref_logps(const TensorPtr& chosen_hidden,
                                            const TensorPtr& rejected_hidden,
                                            const TensorPtr& lm_head_weight,
                                            const std::vector<float>& ref_chosen_logps,
                                            const std::vector<float>& ref_rejected_logps,
                                            const TensorPtr& chosen_input_ids,
                                            const TensorPtr& rejected_input_ids,
                                            const TensorPtr& chosen_response_mask,
                                            const TensorPtr& rejected_response_mask,
                                            float beta) {
    auto policy_chosen_logps = sequence_logprob_masked_from_hidden(
        chosen_hidden, lm_head_weight, chosen_input_ids, chosen_response_mask);
    auto policy_rejected_logps = sequence_logprob_masked_from_hidden(
        rejected_hidden, lm_head_weight, rejected_input_ids, rejected_response_mask);

    auto computed = compute_loss_and_coeffs(policy_chosen_logps, policy_rejected_logps,
                                            ref_chosen_logps, ref_rejected_logps,
                                            beta, "streaming_dpo_loss_with_ref_logps");
    auto out = full({1}, computed.first, kFloat32, kCPU);
    if (chosen_hidden->requires_grad() || rejected_hidden->requires_grad() ||
        lm_head_weight->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<StreamingDPOBackward>(
            chosen_hidden, rejected_hidden, lm_head_weight,
            chosen_input_ids, rejected_input_ids,
            chosen_response_mask, rejected_response_mask,
            std::move(computed.second.first), std::move(computed.second.second));
#ifdef USE_NEW_AUTOGRAD_ENGINE
        autograd::Engine::instance().register_node(
            out,
            {chosen_hidden, rejected_hidden, lm_head_weight,
             chosen_input_ids, rejected_input_ids, chosen_response_mask, rejected_response_mask},
            backward_fn);
#else
        out->set_grad_fn([backward_fn, chosen_hidden, rejected_hidden, lm_head_weight](const TensorPtr& grad_output)
                             -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (chosen_hidden->requires_grad() && grads[0]) accumulate_gradient(chosen_hidden, grads[0]);
            if (rejected_hidden->requires_grad() && grads[1]) accumulate_gradient(rejected_hidden, grads[1]);
            if (lm_head_weight->requires_grad() && grads[2]) accumulate_gradient(lm_head_weight, grads[2]);
            return grads;
        });
#endif
    }
    return out;
}

DPOMetrics compute_dpo_metrics_from_logps(const std::vector<float>& policy_chosen_logps,
                                          const std::vector<float>& policy_rejected_logps,
                                          const std::vector<float>& ref_chosen_logps,
                                          const std::vector<float>& ref_rejected_logps,
                                          float beta) {
    validate_dpo_logps(policy_chosen_logps, policy_rejected_logps,
                       ref_chosen_logps, ref_rejected_logps,
                       "compute_dpo_metrics_from_logps");
    if (!(beta > 0.0f)) {
        throw std::runtime_error("compute_dpo_metrics_from_logps: beta must be positive");
    }

    DPOMetrics metrics;
    const size_t n = policy_chosen_logps.size();
    double loss_sum = 0.0;
    double chosen_reward_sum = 0.0;
    double rejected_reward_sum = 0.0;
    double reward_margin_sum = 0.0;
    double reward_accuracy_sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double chosen_reward =
            static_cast<double>(beta) *
            (static_cast<double>(policy_chosen_logps[i]) -
             static_cast<double>(ref_chosen_logps[i]));
        const double rejected_reward =
            static_cast<double>(beta) *
            (static_cast<double>(policy_rejected_logps[i]) -
             static_cast<double>(ref_rejected_logps[i]));
        const double margin = chosen_reward - rejected_reward;

        loss_sum += neg_log_sigmoid(margin);
        chosen_reward_sum += chosen_reward;
        rejected_reward_sum += rejected_reward;
        reward_margin_sum += margin;
        reward_accuracy_sum += margin > 0.0 ? 1.0 : 0.0;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    metrics.loss = static_cast<float>(loss_sum * inv_n);
    metrics.chosen_reward = static_cast<float>(chosen_reward_sum * inv_n);
    metrics.rejected_reward = static_cast<float>(rejected_reward_sum * inv_n);
    metrics.reward_margin = static_cast<float>(reward_margin_sum * inv_n);
    metrics.reward_accuracy = static_cast<float>(reward_accuracy_sum * inv_n);
    return metrics;
}

DPOMetrics compute_dpo_metrics_from_logits(const TensorPtr& policy_chosen_logits,
                                           const TensorPtr& policy_rejected_logits,
                                           const TensorPtr& ref_chosen_logits,
                                           const TensorPtr& ref_rejected_logits,
                                           const TensorPtr& chosen_input_ids,
                                           const TensorPtr& rejected_input_ids,
                                           const TensorPtr& chosen_response_mask,
                                           const TensorPtr& rejected_response_mask,
                                           float beta) {
    return compute_dpo_metrics_from_logps(
        sequence_logprob_masked(policy_chosen_logits, chosen_input_ids, chosen_response_mask),
        sequence_logprob_masked(policy_rejected_logits, rejected_input_ids, rejected_response_mask),
        sequence_logprob_masked(ref_chosen_logits, chosen_input_ids, chosen_response_mask),
        sequence_logprob_masked(ref_rejected_logits, rejected_input_ids, rejected_response_mask),
        beta);
}

DPOMetrics compute_dpo_metrics_from_policy_logits_and_ref_logps(
    const TensorPtr& policy_chosen_logits,
    const TensorPtr& policy_rejected_logits,
    const std::vector<float>& ref_chosen_logps,
    const std::vector<float>& ref_rejected_logps,
    const TensorPtr& chosen_input_ids,
    const TensorPtr& rejected_input_ids,
    const TensorPtr& chosen_response_mask,
    const TensorPtr& rejected_response_mask,
    float beta) {
    return compute_dpo_metrics_from_logps(
        sequence_logprob_masked(policy_chosen_logits, chosen_input_ids, chosen_response_mask),
        sequence_logprob_masked(policy_rejected_logits, rejected_input_ids, rejected_response_mask),
        ref_chosen_logps,
        ref_rejected_logps,
        beta);
}

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
    float beta) {
    return compute_dpo_metrics_from_logps(
        sequence_logprob_masked_from_hidden(chosen_hidden, lm_head_weight,
                                            chosen_input_ids, chosen_response_mask),
        sequence_logprob_masked_from_hidden(rejected_hidden, lm_head_weight,
                                            rejected_input_ids, rejected_response_mask),
        ref_chosen_logps,
        ref_rejected_logps,
        beta);
}

}  // namespace ops
