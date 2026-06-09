#include "causal_lm_batch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ops {

namespace {

void validate_config(const CausalLMBatchConfig& config) {
    if (config.sequence_length <= 1) {
        throw std::invalid_argument("CausalLMBatchConfig.sequence_length must be > 1");
    }
}

int resolve_pad_token(const Tokenizer& tokenizer) {
    int pad = tokenizer.get_pad_token();
    if (pad < 0) {
        pad = tokenizer.get_eos_token();
    }
    return pad >= 0 ? pad : 0;
}

EncodedInput encode_for_causal_lm_batch(Tokenizer& tokenizer,
                                        const std::string& text,
                                        const CausalLMBatchConfig& config) {
    TokenizerEncodeOptions options;
    options.add_special_tokens = tokenizer.default_add_special_tokens();
    options.max_length = 0;
    options.truncation = false;

    std::vector<int> ids = tokenizer.encode_with_options(text, options);
    if (config.append_eos) {
        const int eos = tokenizer.get_eos_token();
        if (eos < 0) {
            throw std::invalid_argument("append_eos=true requires tokenizer EOS token");
        }
        if (ids.empty() || ids.back() != eos) {
            ids.push_back(eos);
        }
    }

    const int seq = config.sequence_length;
    if (config.truncate && static_cast<int>(ids.size()) > seq) {
        ids.resize(seq);
    }
    if (static_cast<int>(ids.size()) > seq) {
        throw std::invalid_argument("encoded text is longer than sequence_length and truncate=false");
    }

    std::vector<int> mask(ids.size(), 1);
    if (static_cast<int>(ids.size()) < seq) {
        const int pad = resolve_pad_token(tokenizer);
        const int pad_count = seq - static_cast<int>(ids.size());
        if (tokenizer.default_left_padding()) {
            ids.insert(ids.begin(), pad_count, pad);
            mask.insert(mask.begin(), pad_count, 0);
        } else {
            ids.insert(ids.end(), pad_count, pad);
            mask.insert(mask.end(), pad_count, 0);
        }
    }

    return {std::move(ids), std::move(mask)};
}

CausalLMBatch make_batch_from_encoded(const std::vector<std::vector<int>>& ids_rows,
                                      const std::vector<std::vector<int>>& mask_rows,
                                      const CausalLMBatchConfig& config) {
    validate_config(config);
    if (ids_rows.empty()) {
        throw std::invalid_argument("make_causal_lm_batch requires at least one text");
    }
    if (ids_rows.size() != mask_rows.size()) {
        throw std::invalid_argument("input_ids and attention_mask row counts differ");
    }

    const int batch = static_cast<int>(ids_rows.size());
    const int seq = config.sequence_length;
    std::vector<int32_t> input_ids(static_cast<size_t>(batch * seq), 0);
    std::vector<float> attention_mask(static_cast<size_t>(batch * seq), 0.0f);
    std::vector<int32_t> labels(static_cast<size_t>(batch * seq), config.ignore_index);
    int valid_labels = 0;

    for (int b = 0; b < batch; ++b) {
        if (static_cast<int>(ids_rows[b].size()) != seq ||
            static_cast<int>(mask_rows[b].size()) != seq) {
            throw std::invalid_argument("encoded rows must already be padded to sequence_length");
        }

        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            const int token_id = ids_rows[b][s];
            const int mask = mask_rows[b][s] != 0 ? 1 : 0;
            input_ids[static_cast<size_t>(idx)] = static_cast<int32_t>(token_id);
            attention_mask[static_cast<size_t>(idx)] = static_cast<float>(mask);

            // lm_cross_entropy consumes labels[s + 1] for logits[s].
            const bool previous_token_is_real =
                s > 0 && (mask_rows[b][s - 1] != 0);
            if (s > 0 &&
                (!config.mask_padding_labels || (mask != 0 && previous_token_is_real))) {
                labels[static_cast<size_t>(idx)] = static_cast<int32_t>(token_id);
                ++valid_labels;
            }
        }
    }

    CausalLMBatch out;
    out.input_ids = std::make_shared<Tensor>(
        std::vector<int64_t>{batch, seq}, input_ids.data(), kInt32, kCPU);
    out.attention_mask = std::make_shared<Tensor>(
        std::vector<int64_t>{batch, seq}, attention_mask.data(), kFloat32, kCPU);
    out.labels = std::make_shared<Tensor>(
        std::vector<int64_t>{batch, seq}, labels.data(), kInt32, kCPU);
    out.batch_size = batch;
    out.sequence_length = seq;
    out.valid_label_count = valid_labels;
    return out;
}

}  // namespace

CausalLMBatch make_causal_lm_batch(Tokenizer& tokenizer,
                                   const std::vector<std::string>& texts,
                                   const CausalLMBatchConfig& config) {
    validate_config(config);
    if (texts.empty()) {
        throw std::invalid_argument("make_causal_lm_batch requires at least one text");
    }

    std::vector<std::vector<int>> ids_rows;
    std::vector<std::vector<int>> mask_rows;
    ids_rows.reserve(texts.size());
    mask_rows.reserve(texts.size());
    for (const auto& text : texts) {
        auto encoded = encode_for_causal_lm_batch(tokenizer, text, config);
        ids_rows.push_back(std::move(encoded.input_ids));
        mask_rows.push_back(std::move(encoded.attention_mask));
    }
    return make_batch_from_encoded(ids_rows, mask_rows, config);
}

CausalLMBatch make_causal_lm_batch_from_token_ids(const std::vector<std::vector<int>>& token_ids,
                                                  int pad_token_id,
                                                  const CausalLMBatchConfig& config) {
    validate_config(config);
    if (token_ids.empty()) {
        throw std::invalid_argument("make_causal_lm_batch_from_token_ids requires at least one row");
    }

    const int seq = config.sequence_length;
    std::vector<std::vector<int>> ids_rows;
    std::vector<std::vector<int>> mask_rows;
    ids_rows.reserve(token_ids.size());
    mask_rows.reserve(token_ids.size());

    for (const auto& row : token_ids) {
        std::vector<int> ids = row;
        if (config.truncate && static_cast<int>(ids.size()) > seq) {
            ids.resize(seq);
        }

        std::vector<int> mask(std::min<int>(static_cast<int>(ids.size()), seq), 1);
        if (static_cast<int>(ids.size()) < seq) {
            const int pad_count = seq - static_cast<int>(ids.size());
            ids.insert(ids.end(), pad_count, pad_token_id);
            mask.insert(mask.end(), pad_count, 0);
        }
        if (static_cast<int>(ids.size()) != seq) {
            throw std::invalid_argument("token row is longer than sequence_length and truncate=false");
        }
        ids_rows.push_back(std::move(ids));
        mask_rows.push_back(std::move(mask));
    }

    return make_batch_from_encoded(ids_rows, mask_rows, config);
}

}  // namespace ops
