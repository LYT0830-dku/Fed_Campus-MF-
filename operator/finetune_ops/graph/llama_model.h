/**
 * @file llama_model.h
 * @brief Llama-family decoder graph (RMSNorm + RoPE + GQA + SwiGLU)
 *
 * This model intentionally lives beside Qwen/Gemma instead of reusing Qwen
 * directly: HF Llama checkpoints commonly use untied lm_head.weight and do not
 * provide q/k/v projection biases.
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ops {

struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    int head_dim = -1;
    int max_position_embeddings = 2048;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = -1;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool has_rope_scaling = false;
    std::string rope_scaling_type;
    float rope_scaling_factor = 1.0f;
    float rope_low_freq_factor = 1.0f;
    float rope_high_freq_factor = 4.0f;
    int rope_original_max_position_embeddings = 0;

    int effective_head_dim() const {
        return head_dim > 0 ? head_dim : hidden_size / num_attention_heads;
    }

    bool uses_llama3_rope_scaling() const {
        return has_rope_scaling && rope_scaling_type == "llama3";
    }

    static LlamaConfig from_pretrained(const std::string& config_path);
};

struct LlamaBlock {
    TensorPtr input_norm_weight;
    TensorPtr post_norm_weight;

    TensorPtr q_proj_weight;
    TensorPtr k_proj_weight;
    TensorPtr v_proj_weight;
    TensorPtr o_proj_weight;

    TensorPtr q_proj_bias;
    TensorPtr k_proj_bias;
    TensorPtr v_proj_bias;
    TensorPtr o_proj_bias;

    TensorPtr gate_proj_weight;
    TensorPtr up_proj_weight;
    TensorPtr down_proj_weight;

    std::unique_ptr<LoRALinear> q_lin;
    std::unique_ptr<LoRALinear> k_lin;
    std::unique_ptr<LoRALinear> v_lin;
    std::unique_ptr<LoRALinear> o_lin;

    bool lora_initialized = false;
};

class LlamaModel {
public:
    explicit LlamaModel(const LlamaConfig& cfg);
    ~LlamaModel() = default;

    TensorPtr forward_hidden(const TensorPtr& input_ids,
                             const TensorPtr& attention_mask = nullptr);
    TensorPtr lm_head(const TensorPtr& hidden);
    TensorPtr lm_head_weight_for_loss() const;
    TensorPtr forward(const TensorPtr& input_ids,
                      const TensorPtr& attention_mask = nullptr);

    void init_lora(int rank = 8, float alpha = 16.0f, float dropout = 0.05f,
                   bool qv_only = false, uint64_t seed = 42);
    std::vector<TensorPtr> get_lora_parameters() const;
    std::vector<std::pair<std::string, TensorPtr>> named_lora_parameters() const;
    std::vector<TensorPtr> parameters() const;
    void freeze_base();

    void assign_weight(const std::string& key,
                       const TensorPtr& tensor,
                       bool strict_shape_check = true);

    const LlamaConfig& config() const { return config_; }
    const TensorPtr& embedding_weight() const { return embed_tokens_; }

private:
    LlamaConfig config_;
    TensorPtr embed_tokens_;       // [vocab, hidden]
    TensorPtr final_norm_weight_;  // [hidden]
    TensorPtr lm_head_weight_;     // [hidden, vocab] when untied

    std::vector<LlamaBlock> layers_;

    TensorPtr embedding_lookup(const TensorPtr& weight,
                               const TensorPtr& indices) const;
    TensorPtr build_causal_mask(int seq_len) const;
    TensorPtr build_padding_mask(const TensorPtr& attention_mask) const;
    TensorPtr attention(const TensorPtr& x,
                        LlamaBlock& block,
                        const TensorPtr& causal_mask,
                        const TensorPtr& pad_mask);
    TensorPtr mlp(const TensorPtr& x, LlamaBlock& block) const;
};

}  // namespace ops
