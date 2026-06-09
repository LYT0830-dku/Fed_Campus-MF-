/**
 * @file llama_model.cpp
 * @brief Llama-family decoder graph implementation.
 */

#include "llama_model.h"

#include "../core/ops.h"
#include "../core/utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ops {

namespace {

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void assign_checked(TensorPtr& slot,
                    const std::string& key,
                    const TensorPtr& tensor,
                    bool strict_shape_check) {
    if (!tensor) {
        throw std::runtime_error("LlamaModel::assign_weight received null tensor for " + key);
    }
    if (strict_shape_check && slot && slot->shape() != tensor->shape()) {
        throw std::runtime_error(
            "LlamaModel::assign_weight shape mismatch for " + key +
            ": expected=" + shape_to_string(slot->shape()) +
            ", actual=" + shape_to_string(tensor->shape()));
    }
    slot = tensor;
}

float weight_value(const TensorPtr& weight, int64_t idx) {
    if (weight->dtype() == kFloat32) {
        return weight->data<float>()[idx];
    }
    if (weight->dtype() == kFloat16) {
        return fp16_bits_to_float32(weight->data<uint16_t>()[idx]);
    }
    if (weight->dtype() == kBFloat16) {
        return bf16_bits_to_float32(weight->data<uint16_t>()[idx]);
    }
    throw std::runtime_error("LlamaModel: unsupported weight dtype " +
                             DTypeUtils::to_string(weight->dtype()));
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config: " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

int json_int(const std::string& json, const std::string& key, int fallback) {
    const std::regex pat("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, pat)) {
        return std::stoi(match[1].str());
    }
    return fallback;
}

float json_float(const std::string& json, const std::string& key, float fallback) {
    const std::regex pat("\"" + key + "\"\\s*:\\s*([0-9eE\\.+-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pat)) {
        return std::stof(match[1].str());
    }
    return fallback;
}

std::string json_string(const std::string& json,
                        const std::string& key,
                        const std::string& fallback) {
    const std::regex pat("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(json, match, pat)) {
        return match[1].str();
    }
    return fallback;
}

bool json_bool(const std::string& json, const std::string& key, bool fallback) {
    const std::regex pat("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pat)) {
        return match[1].str() == "true";
    }
    return fallback;
}

bool json_has_key(const std::string& json, const std::string& key) {
    const std::regex pat("\"" + key + "\"\\s*:");
    return std::regex_search(json, pat);
}

}  // namespace

LlamaConfig LlamaConfig::from_pretrained(const std::string& config_path) {
    LlamaConfig cfg;
    const std::string json = read_file(config_path);
    cfg.vocab_size = json_int(json, "vocab_size", cfg.vocab_size);
    cfg.hidden_size = json_int(json, "hidden_size", cfg.hidden_size);
    cfg.intermediate_size = json_int(json, "intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = json_int(json, "num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = json_int(json, "num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = json_int(json, "num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim = json_int(json, "head_dim", cfg.head_dim);
    cfg.max_position_embeddings =
        json_int(json, "max_position_embeddings", cfg.max_position_embeddings);
    cfg.bos_token_id = json_int(json, "bos_token_id", cfg.bos_token_id);
    cfg.eos_token_id = json_int(json, "eos_token_id", cfg.eos_token_id);
    cfg.pad_token_id = json_int(json, "pad_token_id", cfg.pad_token_id);
    cfg.rms_norm_eps = json_float(json, "rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta = json_float(json, "rope_theta", cfg.rope_theta);
    cfg.hidden_act = json_string(json, "hidden_act", cfg.hidden_act);
    cfg.tie_word_embeddings = json_bool(json, "tie_word_embeddings", cfg.tie_word_embeddings);
    cfg.attention_bias = json_bool(json, "attention_bias", cfg.attention_bias);
    cfg.has_rope_scaling = json_has_key(json, "rope_scaling");
    if (cfg.has_rope_scaling) {
        cfg.rope_scaling_type = json_string(json, "rope_type", "");
        if (cfg.rope_scaling_type.empty()) {
            cfg.rope_scaling_type = json_string(json, "type", "");
        }
        cfg.rope_scaling_factor =
            json_float(json, "factor", cfg.rope_scaling_factor);
        cfg.rope_low_freq_factor =
            json_float(json, "low_freq_factor", cfg.rope_low_freq_factor);
        cfg.rope_high_freq_factor =
            json_float(json, "high_freq_factor", cfg.rope_high_freq_factor);
        cfg.rope_original_max_position_embeddings =
            json_int(json, "original_max_position_embeddings",
                     cfg.rope_original_max_position_embeddings);
    }
    return cfg;
}

LlamaModel::LlamaModel(const LlamaConfig& cfg) : config_(cfg) {
    if (config_.num_attention_heads <= 0 ||
        config_.num_key_value_heads <= 0 ||
        config_.hidden_size <= 0 ||
        config_.effective_head_dim() <= 0) {
        throw std::runtime_error("LlamaModel: invalid attention dimensions");
    }
    if (config_.num_attention_heads % config_.num_key_value_heads != 0) {
        throw std::runtime_error("LlamaModel: attention heads must be divisible by KV heads");
    }
    if (config_.hidden_size != config_.num_attention_heads * config_.effective_head_dim()) {
        throw std::runtime_error("LlamaModel: hidden_size must equal num_attention_heads * head_dim");
    }
    if (config_.hidden_act != "silu") {
        throw std::runtime_error("LlamaModel: only hidden_act=silu is currently supported");
    }
    if (config_.has_rope_scaling && !config_.uses_llama3_rope_scaling()) {
        throw std::runtime_error(
            "LlamaModel: unsupported rope_scaling type '" +
            config_.rope_scaling_type + "'; only rope_type=llama3 is supported");
    }
    if (config_.uses_llama3_rope_scaling() &&
        config_.rope_original_max_position_embeddings <= 0) {
        throw std::runtime_error(
            "LlamaModel: Llama3 rope_scaling requires original_max_position_embeddings");
    }

    embed_tokens_ =
        std::make_shared<Tensor>(std::vector<int64_t>{config_.vocab_size, config_.hidden_size},
                                 kFloat32, kCPU);
    final_norm_weight_ =
        std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
    if (!config_.tie_word_embeddings) {
        lm_head_weight_ =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size, config_.vocab_size},
                                     kFloat32, kCPU);
    }

    layers_.resize(config_.num_hidden_layers);
    const int64_t kv_out =
        static_cast<int64_t>(config_.num_key_value_heads) * config_.effective_head_dim();
    for (auto& block : layers_) {
        block.input_norm_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
        block.post_norm_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);

        block.q_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size, config_.hidden_size},
                                     kFloat32, kCPU);
        block.k_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size, kv_out},
                                     kFloat32, kCPU);
        block.v_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size, kv_out},
                                     kFloat32, kCPU);
        block.o_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size, config_.hidden_size},
                                     kFloat32, kCPU);

        if (config_.attention_bias) {
            block.q_proj_bias =
                std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
            block.k_proj_bias =
                std::make_shared<Tensor>(std::vector<int64_t>{kv_out}, kFloat32, kCPU);
            block.v_proj_bias =
                std::make_shared<Tensor>(std::vector<int64_t>{kv_out}, kFloat32, kCPU);
            block.o_proj_bias =
                std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size}, kFloat32, kCPU);
        }

        block.gate_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size,
                                                          config_.intermediate_size},
                                     kFloat32, kCPU);
        block.up_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.hidden_size,
                                                          config_.intermediate_size},
                                     kFloat32, kCPU);
        block.down_proj_weight =
            std::make_shared<Tensor>(std::vector<int64_t>{config_.intermediate_size,
                                                          config_.hidden_size},
                                     kFloat32, kCPU);
    }
}

void LlamaModel::assign_weight(const std::string& key,
                               const TensorPtr& tensor,
                               bool strict_shape_check) {
    if (key == "embed_tokens.weight") {
        assign_checked(embed_tokens_, key, tensor, strict_shape_check);
        return;
    }
    if (key == "final_norm.weight") {
        assign_checked(final_norm_weight_, key, tensor, strict_shape_check);
        return;
    }
    if (key == "lm_head.weight") {
        if (config_.tie_word_embeddings) {
            throw std::runtime_error("LlamaModel: lm_head.weight received for tied checkpoint");
        }
        assign_checked(lm_head_weight_, key, tensor, strict_shape_check);
        return;
    }

    std::regex layer_key(R"(layers\.(\d+)\.(.+))");
    std::smatch match;
    if (!std::regex_match(key, match, layer_key)) {
        return;
    }

    const int idx = std::stoi(match[1].str());
    const std::string sub = match[2].str();
    if (idx < 0 || idx >= config_.num_hidden_layers) {
        throw std::runtime_error("LlamaModel::assign_weight bad layer index");
    }
    auto& block = layers_[idx];
    if (sub == "input_norm.weight") {
        assign_checked(block.input_norm_weight, key, tensor, strict_shape_check);
    } else if (sub == "post_norm.weight") {
        assign_checked(block.post_norm_weight, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.q_proj.weight") {
        assign_checked(block.q_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.k_proj.weight") {
        assign_checked(block.k_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.v_proj.weight") {
        assign_checked(block.v_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.o_proj.weight") {
        assign_checked(block.o_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.q_proj.bias") {
        assign_checked(block.q_proj_bias, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.k_proj.bias") {
        assign_checked(block.k_proj_bias, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.v_proj.bias") {
        assign_checked(block.v_proj_bias, key, tensor, strict_shape_check);
    } else if (sub == "self_attn.o_proj.bias") {
        assign_checked(block.o_proj_bias, key, tensor, strict_shape_check);
    } else if (sub == "mlp.gate_proj.weight") {
        assign_checked(block.gate_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "mlp.up_proj.weight") {
        assign_checked(block.up_proj_weight, key, tensor, strict_shape_check);
    } else if (sub == "mlp.down_proj.weight") {
        assign_checked(block.down_proj_weight, key, tensor, strict_shape_check);
    }
}

void LlamaModel::init_lora(int rank,
                           float alpha,
                           float dropout [[maybe_unused]],
                           bool qv_only,
                           uint64_t seed) {
    if (rank <= 0) {
        throw std::runtime_error("LlamaModel::init_lora rank must be positive");
    }
    const float scale = alpha / static_cast<float>(rank);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> init_dist(-0.01f, 0.01f);

    auto make_ab = [&](int64_t in_dim, int64_t out_dim) -> std::pair<TensorPtr, TensorPtr> {
        auto A = std::make_shared<Tensor>(std::vector<int64_t>{in_dim, rank}, kFloat32, kCPU);
        auto B = std::make_shared<Tensor>(std::vector<int64_t>{rank, out_dim}, kFloat32, kCPU);
        float* a_data = A->data<float>();
        for (int64_t i = 0; i < in_dim * rank; ++i) {
            a_data[i] = init_dist(rng);
        }
        std::memset(B->data<void>(), 0, sizeof(float) * static_cast<size_t>(rank) *
                                           static_cast<size_t>(out_dim));
        A->set_requires_grad(true);
        B->set_requires_grad(true);
        return {A, B};
    };

    const int64_t kv_out =
        static_cast<int64_t>(config_.num_key_value_heads) * config_.effective_head_dim();
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        auto& block = layers_[layer_idx];
        if (block.lora_initialized) {
            continue;
        }
        block.q_lin = std::make_unique<LoRALinear>(block.q_proj_weight, block.q_proj_bias);
        block.v_lin = std::make_unique<LoRALinear>(block.v_proj_weight, block.v_proj_bias);
        block.q_lin->set_debug_name("layers." + std::to_string(layer_idx) + ".self_attn.q_proj");
        block.v_lin->set_debug_name("layers." + std::to_string(layer_idx) + ".self_attn.v_proj");

        auto [Aq, Bq] = make_ab(config_.hidden_size, config_.hidden_size);
        auto [Av, Bv] = make_ab(config_.hidden_size, kv_out);
        block.q_lin->attach_lora(Aq, Bq, scale);
        block.v_lin->attach_lora(Av, Bv, scale);

        if (!qv_only) {
            block.k_lin = std::make_unique<LoRALinear>(block.k_proj_weight, block.k_proj_bias);
            block.o_lin = std::make_unique<LoRALinear>(block.o_proj_weight, block.o_proj_bias);
            block.k_lin->set_debug_name("layers." + std::to_string(layer_idx) + ".self_attn.k_proj");
            block.o_lin->set_debug_name("layers." + std::to_string(layer_idx) + ".self_attn.o_proj");
            auto [Ak, Bk] = make_ab(config_.hidden_size, kv_out);
            auto [Ao, Bo] = make_ab(config_.hidden_size, config_.hidden_size);
            block.k_lin->attach_lora(Ak, Bk, scale);
            block.o_lin->attach_lora(Ao, Bo, scale);
        }
        block.lora_initialized = true;
    }
}

std::vector<TensorPtr> LlamaModel::get_lora_parameters() const {
    std::vector<TensorPtr> params;
    for (const auto& block : layers_) {
        if (!block.lora_initialized) {
            continue;
        }
        auto add = [&](const std::unique_ptr<LoRALinear>& lin) {
            if (!lin || lin->slices().empty()) {
                return;
            }
            auto local = lin->trainable_parameters();
            params.insert(params.end(), local.begin(), local.end());
        };
        add(block.q_lin);
        add(block.k_lin);
        add(block.v_lin);
        add(block.o_lin);
    }
    return params;
}

std::vector<std::pair<std::string, TensorPtr>> LlamaModel::named_lora_parameters() const {
    std::vector<std::pair<std::string, TensorPtr>> params;
    for (const auto& block : layers_) {
        if (!block.lora_initialized) {
            continue;
        }
        auto add = [&](const std::unique_ptr<LoRALinear>& lin) {
            if (!lin || lin->slices().empty()) {
                return;
            }
            auto local = lin->debug_params();
            params.insert(params.end(), local.begin(), local.end());
        };
        add(block.q_lin);
        add(block.k_lin);
        add(block.v_lin);
        add(block.o_lin);
    }
    return params;
}

std::vector<TensorPtr> LlamaModel::parameters() const {
    std::vector<TensorPtr> params;
    params.reserve(3 + static_cast<size_t>(config_.num_hidden_layers) *
                           (config_.attention_bias ? 13 : 9));
    params.push_back(embed_tokens_);
    params.push_back(final_norm_weight_);
    if (!config_.tie_word_embeddings) {
        params.push_back(lm_head_weight_);
    }
    for (const auto& block : layers_) {
        params.push_back(block.input_norm_weight);
        params.push_back(block.post_norm_weight);
        params.push_back(block.q_proj_weight);
        params.push_back(block.k_proj_weight);
        params.push_back(block.v_proj_weight);
        params.push_back(block.o_proj_weight);
        if (config_.attention_bias) {
            params.push_back(block.q_proj_bias);
            params.push_back(block.k_proj_bias);
            params.push_back(block.v_proj_bias);
            params.push_back(block.o_proj_bias);
        }
        params.push_back(block.gate_proj_weight);
        params.push_back(block.up_proj_weight);
        params.push_back(block.down_proj_weight);
    }
    return params;
}

void LlamaModel::freeze_base() {
    auto freeze = [](const TensorPtr& t) {
        if (t) {
            t->set_requires_grad(false);
        }
    };
    freeze(embed_tokens_);
    freeze(final_norm_weight_);
    freeze(lm_head_weight_);
    for (auto& block : layers_) {
        freeze(block.input_norm_weight);
        freeze(block.post_norm_weight);
        freeze(block.q_proj_weight);
        freeze(block.k_proj_weight);
        freeze(block.v_proj_weight);
        freeze(block.o_proj_weight);
        freeze(block.q_proj_bias);
        freeze(block.k_proj_bias);
        freeze(block.v_proj_bias);
        freeze(block.o_proj_bias);
        freeze(block.gate_proj_weight);
        freeze(block.up_proj_weight);
        freeze(block.down_proj_weight);
    }
}

TensorPtr LlamaModel::embedding_lookup(const TensorPtr& weight,
                                       const TensorPtr& indices) const {
    if (!indices || indices->ndim() != 2 || indices->dtype() != kInt32) {
        throw std::runtime_error("LlamaModel::embedding_lookup expects [B,S] int32 input_ids");
    }
    const auto& shape = indices->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t H = weight->shape()[1];
    auto out = zeros({B, S, H}, kFloat32, kCPU);
    const int32_t* ids = indices->data<int32_t>();
    float* dst = out->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            const int32_t id = ids[b * S + s];
            if (id < 0 || id >= config_.vocab_size) {
                throw std::runtime_error("LlamaModel::embedding_lookup token id out of range");
            }
            float* row = dst + (b * S + s) * H;
            const int64_t base = static_cast<int64_t>(id) * H;
            if (weight->dtype() == kFloat32) {
                std::memcpy(row, weight->data<float>() + base, sizeof(float) * static_cast<size_t>(H));
            } else {
                for (int64_t h = 0; h < H; ++h) {
                    row[h] = weight_value(weight, base + h);
                }
            }
        }
    }
    return out;
}

TensorPtr LlamaModel::build_causal_mask(int seq_len) const {
    return ops::create_causal_mask(seq_len, kFloat32, kCPU);
}

TensorPtr LlamaModel::build_padding_mask(const TensorPtr& attention_mask) const {
    if (!attention_mask) {
        return nullptr;
    }
    if (attention_mask->ndim() != 2 || attention_mask->dtype() != kFloat32) {
        throw std::runtime_error("LlamaModel::build_padding_mask expects [B,S] float32 attention_mask");
    }
    const int64_t B = attention_mask->shape()[0];
    const int64_t S = attention_mask->shape()[1];
    auto mask = zeros({B, 1, 1, S}, kFloat32, kCPU);
    const float* src = attention_mask->data<float>();
    float* dst = mask->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            dst[b * S + s] = src[b * S + s] > 0.5f ? 0.0f : -1e9f;
        }
    }
    return mask;
}

TensorPtr LlamaModel::attention(const TensorPtr& x,
                                LlamaBlock& block,
                                const TensorPtr& causal_mask,
                                const TensorPtr& pad_mask) {
    const int64_t B = x->shape()[0];
    const int64_t S = x->shape()[1];
    const int64_t C = x->shape()[2];
    const int64_t H = config_.num_attention_heads;
    const int64_t HKV = config_.num_key_value_heads;
    const int64_t Hd = config_.effective_head_dim();

    auto q = block.lora_initialized && block.q_lin
        ? block.q_lin->forward(x)
        : (block.q_proj_bias ? add(matmul(x, block.q_proj_weight), block.q_proj_bias)
                             : matmul(x, block.q_proj_weight));
    auto k = block.lora_initialized && block.k_lin
        ? block.k_lin->forward(x)
        : (block.k_proj_bias ? add(matmul(x, block.k_proj_weight), block.k_proj_bias)
                             : matmul(x, block.k_proj_weight));
    auto v = block.lora_initialized && block.v_lin
        ? block.v_lin->forward(x)
        : (block.v_proj_bias ? add(matmul(x, block.v_proj_weight), block.v_proj_bias)
                             : matmul(x, block.v_proj_weight));

    q = reshape(q, {B, S, H, Hd});
    q = permute(q, {0, 2, 1, 3});
    k = reshape(k, {B, S, HKV, Hd});
    v = reshape(v, {B, S, HKV, Hd});
    k = permute(k, {0, 2, 1, 3});
    v = permute(v, {0, 2, 1, 3});

    if (HKV != H) {
        const int repeat = static_cast<int>(H / HKV);
        k = repeat_kv_heads(k, repeat);
        v = repeat_kv_heads(v, repeat);
    }

    q = apply_rope(q, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta,
                   config_.uses_llama3_rope_scaling(), config_.rope_scaling_factor,
                   config_.rope_low_freq_factor, config_.rope_high_freq_factor,
                   config_.rope_original_max_position_embeddings);
    k = apply_rope(k, static_cast<int>(S), static_cast<int>(Hd), config_.rope_theta,
                   config_.uses_llama3_rope_scaling(), config_.rope_scaling_factor,
                   config_.rope_low_freq_factor, config_.rope_high_freq_factor,
                   config_.rope_original_max_position_embeddings);

    auto scores = matmul(q, transpose(k, 2, 3));
    scores = mul(scores, 1.0f / std::sqrt(static_cast<float>(Hd)));
    if (causal_mask) {
        scores = add(scores, causal_mask);
    }
    if (pad_mask) {
        scores = add(scores, pad_mask);
    }
    auto probs = softmax(scores, -1);
    auto ctx = matmul(probs, v);
    ctx = permute(ctx, {0, 2, 1, 3});
    ctx = reshape(ctx, {B, S, C});

    return block.lora_initialized && block.o_lin
        ? block.o_lin->forward(ctx)
        : (block.o_proj_bias ? add(matmul(ctx, block.o_proj_weight), block.o_proj_bias)
                             : matmul(ctx, block.o_proj_weight));
}

TensorPtr LlamaModel::mlp(const TensorPtr& x, LlamaBlock& block) const {
    auto gate = matmul(x, block.gate_proj_weight);
    auto up = matmul(x, block.up_proj_weight);
    auto act = swiglu(gate, up);
    return matmul(act, block.down_proj_weight);
}

TensorPtr LlamaModel::forward_hidden(const TensorPtr& input_ids,
                                     const TensorPtr& attention_mask) {
    auto hidden = embedding_lookup(embed_tokens_, input_ids);
    auto causal_mask = build_causal_mask(static_cast<int>(hidden->shape()[1]));
    auto pad_mask = build_padding_mask(attention_mask);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        auto& block = layers_[i];
        auto normed = rms_norm_affine(hidden, block.input_norm_weight, config_.rms_norm_eps);
        auto attn_out = attention(normed, block, causal_mask, pad_mask);
        hidden = add(hidden, attn_out);

        auto normed2 = rms_norm_affine(hidden, block.post_norm_weight, config_.rms_norm_eps);
        auto mlp_out = mlp(normed2, block);
        hidden = add(hidden, mlp_out);
    }
    return rms_norm_affine(hidden, final_norm_weight_, config_.rms_norm_eps);
}

TensorPtr LlamaModel::lm_head(const TensorPtr& hidden) {
    if (config_.tie_word_embeddings) {
        return matmul_rhs_T(hidden, embed_tokens_);
    }
    if (!lm_head_weight_) {
        throw std::runtime_error("LlamaModel::lm_head missing untied lm_head.weight");
    }
    return matmul(hidden, lm_head_weight_);
}

TensorPtr LlamaModel::lm_head_weight_for_loss() const {
    if (config_.tie_word_embeddings) {
        return embed_tokens_;
    }
    if (!lm_head_weight_) {
        throw std::runtime_error("LlamaModel::lm_head_weight_for_loss missing lm_head.weight");
    }
    return transpose(lm_head_weight_, 0, 1);
}

TensorPtr LlamaModel::forward(const TensorPtr& input_ids,
                              const TensorPtr& attention_mask) {
    return lm_head(forward_hidden(input_ids, attention_mask));
}

}  // namespace ops
