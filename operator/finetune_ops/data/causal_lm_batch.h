#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer.h"

#include <string>
#include <vector>

namespace ops {

struct CausalLMBatchConfig {
    int sequence_length = 64;
    int ignore_index = -100;
    bool truncate = true;
    bool mask_padding_labels = true;
    bool append_eos = false;
};

struct CausalLMBatch {
    TensorPtr input_ids;       // int32 [B, S]
    TensorPtr attention_mask;  // float32 [B, S], 1.0 for real tokens, 0.0 for padding
    TensorPtr labels;          // int32 [B, S], shifted inside lm_cross_entropy

    int batch_size = 0;
    int sequence_length = 0;
    int valid_label_count = 0;
};

/**
 * Build the standard causal-LM full-token batch used by AutoTrainer.
 *
 * Labels are aligned with lm_cross_entropy's HuggingFace-style shift:
 * logits[:, :-1] is compared against labels[:, 1:]. Therefore labels at
 * position 0 are set to ignore_index, and non-padding labels at positions
 * 1..S-1 are the input token IDs. Padding positions are ignored. When
 * append_eos is true, one tokenizer EOS token is appended before truncation
 * and padding so independent text rows can supervise the sample boundary.
 */
CausalLMBatch make_causal_lm_batch(Tokenizer& tokenizer,
                                   const std::vector<std::string>& texts,
                                   const CausalLMBatchConfig& config = CausalLMBatchConfig());

/**
 * Lower-level helper for tests and pre-tokenized datasets.
 */
CausalLMBatch make_causal_lm_batch_from_token_ids(const std::vector<std::vector<int>>& token_ids,
                                                  int pad_token_id,
                                                  const CausalLMBatchConfig& config = CausalLMBatchConfig());

}  // namespace ops
