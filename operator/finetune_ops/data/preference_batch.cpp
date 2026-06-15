#include "preference_batch.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ops {

namespace {

void validate_config(const PreferenceBatchConfig& config) {
    if (config.sequence_length <= 1) {
        throw std::invalid_argument("PreferenceBatchConfig.sequence_length must be > 1");
    }
}

int resolve_pad_token(const Tokenizer& tokenizer) {
    int pad = tokenizer.get_pad_token();
    if (pad < 0) {
        pad = tokenizer.get_eos_token();
    }
    return pad >= 0 ? pad : 0;
}

void append_utf8_codepoint(std::string& out, unsigned int cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

std::size_t skip_ws(const std::string& s, std::size_t pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    return pos;
}

bool parse_json_string_literal(const std::string& s,
                               std::size_t quote_start,
                               std::string& out,
                               std::size_t* next_pos = nullptr) {
    if (quote_start >= s.size() || s[quote_start] != '"') {
        return false;
    }

    std::string value;
    for (std::size_t i = quote_start + 1; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '"') {
            out = std::move(value);
            if (next_pos) {
                *next_pos = i + 1;
            }
            return true;
        }
        if (c != '\\') {
            value.push_back(c);
            continue;
        }

        if (++i >= s.size()) {
            return false;
        }
        const char esc = s[i];
        switch (esc) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u': {
                if (i + 4 >= s.size()) {
                    return false;
                }
                unsigned int cp = 0;
                for (int k = 0; k < 4; ++k) {
                    const int hv = hex_value(s[i + 1 + k]);
                    if (hv < 0) {
                        return false;
                    }
                    cp = (cp << 4) | static_cast<unsigned int>(hv);
                }
                append_utf8_codepoint(value, cp);
                i += 4;
                break;
            }
            default:
                value.push_back(esc);
                break;
        }
    }
    return false;
}

bool find_json_value(const std::string& line,
                     const std::string& key,
                     std::size_t& value_pos) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }
    const std::size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    value_pos = skip_ws(line, colon + 1);
    return value_pos < line.size();
}

bool extract_json_string_field(const std::string& line,
                               const std::string& key,
                               std::string& out) {
    std::size_t value_pos = 0;
    if (!find_json_value(line, key, value_pos)) {
        return false;
    }
    return parse_json_string_literal(line, value_pos, out);
}

bool extract_json_number_field(const std::string& line,
                               const std::string& key,
                               float& out) {
    std::size_t value_pos = 0;
    if (!find_json_value(line, key, value_pos)) {
        return false;
    }
    char* end = nullptr;
    const double parsed = std::strtod(line.c_str() + value_pos, &end);
    if (end == line.c_str() + value_pos || !std::isfinite(parsed)) {
        return false;
    }
    out = static_cast<float>(parsed);
    return true;
}

bool parse_preference_jsonl_row(const std::string& line, PreferenceSample& sample) {
    if (!extract_json_string_field(line, "prompt", sample.prompt) ||
        !extract_json_string_field(line, "chosen", sample.chosen) ||
        !extract_json_string_field(line, "rejected", sample.rejected)) {
        return false;
    }

    float ref_chosen = 0.0f;
    float ref_rejected = 0.0f;
    const bool has_ref_chosen =
        extract_json_number_field(line, "ref_chosen_logp", ref_chosen) ||
        extract_json_number_field(line, "reference_chosen_logp", ref_chosen);
    const bool has_ref_rejected =
        extract_json_number_field(line, "ref_rejected_logp", ref_rejected) ||
        extract_json_number_field(line, "reference_rejected_logp", ref_rejected);
    sample.has_reference_logps = has_ref_chosen && has_ref_rejected;
    if (sample.has_reference_logps) {
        sample.ref_chosen_logp = ref_chosen;
        sample.ref_rejected_logp = ref_rejected;
    }
    return true;
}

std::vector<int> encode_segment(Tokenizer& tokenizer,
                                const std::string& text,
                                const PreferenceBatchConfig& config) {
    TokenizerEncodeOptions options;
    options.add_special_tokens = config.add_special_tokens;
    options.max_length = 0;
    options.truncation = false;
    return tokenizer.encode_with_options(text, options);
}

bool encode_one_branch(Tokenizer& tokenizer,
                       const std::string& prompt,
                       const std::string& response,
                       const PreferenceBatchConfig& config,
                       std::vector<int32_t>& input_ids,
                       std::vector<int32_t>& attention_mask,
                       std::vector<int32_t>& response_mask) {
    std::vector<int> prompt_ids = encode_segment(tokenizer, prompt, config);
    std::vector<int> response_ids = encode_segment(tokenizer, response, config);
    if (config.append_eos_to_response) {
        const int eos = tokenizer.get_eos_token();
        if (eos < 0) {
            throw std::invalid_argument("append_eos_to_response=true requires tokenizer EOS token");
        }
        if (response_ids.empty() || response_ids.back() != eos) {
            response_ids.push_back(eos);
        }
    }
    if (response_ids.empty()) {
        return false;
    }

    std::vector<int32_t> ids;
    std::vector<int32_t> resp;
    ids.reserve(prompt_ids.size() + response_ids.size());
    resp.reserve(prompt_ids.size() + response_ids.size());

    for (int id : prompt_ids) {
        ids.push_back(static_cast<int32_t>(id));
        resp.push_back(0);
    }
    for (int id : response_ids) {
        ids.push_back(static_cast<int32_t>(id));
        resp.push_back(1);
    }

    const int seq = config.sequence_length;
    if (config.truncate && static_cast<int>(ids.size()) > seq) {
        ids.resize(static_cast<std::size_t>(seq));
        resp.resize(static_cast<std::size_t>(seq));
    }
    if (static_cast<int>(ids.size()) > seq) {
        throw std::invalid_argument("preference sample exceeds sequence_length and truncate=false");
    }

    int valid_response_tokens = 0;
    for (int s = 1; s < static_cast<int>(resp.size()); ++s) {
        if (resp[static_cast<std::size_t>(s)] != 0) {
            ++valid_response_tokens;
        }
    }
    if (valid_response_tokens == 0) {
        return false;
    }

    const int pad = resolve_pad_token(tokenizer);
    input_ids.assign(static_cast<std::size_t>(seq), static_cast<int32_t>(pad));
    attention_mask.assign(static_cast<std::size_t>(seq), 0);
    response_mask.assign(static_cast<std::size_t>(seq), 0);

    const int len = static_cast<int>(ids.size());
    const int offset = tokenizer.default_left_padding() ? (seq - len) : 0;
    for (int i = 0; i < len; ++i) {
        const int dst = offset + i;
        input_ids[static_cast<std::size_t>(dst)] = ids[static_cast<std::size_t>(i)];
        attention_mask[static_cast<std::size_t>(dst)] = 1;
        response_mask[static_cast<std::size_t>(dst)] = resp[static_cast<std::size_t>(i)];
    }
    return true;
}

int count_valid_response_tokens(const std::vector<int32_t>& mask) {
    int count = 0;
    for (std::size_t s = 1; s < mask.size(); ++s) {
        if (mask[s] != 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace

bool PreferenceBatch::has_cached_reference_logps() const {
    return static_cast<int>(ref_chosen_logps.size()) == batch_size &&
           static_cast<int>(ref_rejected_logps.size()) == batch_size &&
           batch_size > 0;
}

PreferenceEncodedSample encode_preference_sample(Tokenizer& tokenizer,
                                                 const PreferenceSample& sample,
                                                 const PreferenceBatchConfig& config) {
    validate_config(config);

    PreferenceEncodedSample encoded;
    const bool chosen_ok = encode_one_branch(
        tokenizer, sample.prompt, sample.chosen, config,
        encoded.chosen_input_ids,
        encoded.chosen_attention_mask,
        encoded.chosen_response_mask);
    const bool rejected_ok = encode_one_branch(
        tokenizer, sample.prompt, sample.rejected, config,
        encoded.rejected_input_ids,
        encoded.rejected_attention_mask,
        encoded.rejected_response_mask);
    if (!chosen_ok || !rejected_ok) {
        return PreferenceEncodedSample{};
    }

    encoded.has_reference_logps = sample.has_reference_logps;
    encoded.ref_chosen_logp = sample.ref_chosen_logp;
    encoded.ref_rejected_logp = sample.ref_rejected_logp;
    return encoded;
}

PreferenceBatch make_preference_batch_from_encoded(
    const std::vector<PreferenceEncodedSample>& samples,
    const std::vector<std::size_t>& sample_indices,
    const PreferenceBatchConfig& config) {
    validate_config(config);
    if (samples.empty()) {
        return PreferenceBatch{};
    }
    if (!sample_indices.empty() && sample_indices.size() != samples.size()) {
        throw std::invalid_argument("preference sample index count mismatch");
    }

    const int batch = static_cast<int>(samples.size());
    const int seq = config.sequence_length;
    const std::vector<int64_t> shape{batch, seq};

    PreferenceBatch out;
    out.chosen_input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.chosen_attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    out.chosen_response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.rejected_input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.rejected_attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    out.rejected_response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.batch_size = batch;
    out.sequence_length = seq;
    out.sample_indices = sample_indices;
    if (out.sample_indices.empty()) {
        out.sample_indices.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            out.sample_indices.push_back(i);
        }
    }

    int32_t* c_ids = out.chosen_input_ids->data<int32_t>();
    float* c_attn = out.chosen_attention_mask->data<float>();
    int32_t* c_resp = out.chosen_response_mask->data<int32_t>();
    int32_t* r_ids = out.rejected_input_ids->data<int32_t>();
    float* r_attn = out.rejected_attention_mask->data<float>();
    int32_t* r_resp = out.rejected_response_mask->data<int32_t>();

    bool all_have_ref = true;
    out.ref_chosen_logps.reserve(samples.size());
    out.ref_rejected_logps.reserve(samples.size());

    for (int b = 0; b < batch; ++b) {
        const auto& sample = samples[static_cast<std::size_t>(b)];
        if (static_cast<int>(sample.chosen_input_ids.size()) != seq ||
            static_cast<int>(sample.rejected_input_ids.size()) != seq) {
            throw std::invalid_argument("encoded preference rows must match sequence_length");
        }
        for (int s = 0; s < seq; ++s) {
            const int idx = b * seq + s;
            c_ids[idx] = sample.chosen_input_ids[static_cast<std::size_t>(s)];
            c_attn[idx] = static_cast<float>(sample.chosen_attention_mask[static_cast<std::size_t>(s)]);
            c_resp[idx] = sample.chosen_response_mask[static_cast<std::size_t>(s)];
            r_ids[idx] = sample.rejected_input_ids[static_cast<std::size_t>(s)];
            r_attn[idx] = static_cast<float>(sample.rejected_attention_mask[static_cast<std::size_t>(s)]);
            r_resp[idx] = sample.rejected_response_mask[static_cast<std::size_t>(s)];
        }
        out.valid_response_token_count += count_valid_response_tokens(sample.chosen_response_mask);
        out.valid_response_token_count += count_valid_response_tokens(sample.rejected_response_mask);

        if (sample.has_reference_logps) {
            out.ref_chosen_logps.push_back(sample.ref_chosen_logp);
            out.ref_rejected_logps.push_back(sample.ref_rejected_logp);
        } else {
            all_have_ref = false;
        }
    }

    if (!all_have_ref) {
        out.ref_chosen_logps.clear();
        out.ref_rejected_logps.clear();
    }
    return out;
}

PreferenceBatch make_preference_batch(Tokenizer& tokenizer,
                                      const std::vector<PreferenceSample>& samples,
                                      const PreferenceBatchConfig& config) {
    validate_config(config);
    if (samples.empty()) {
        return PreferenceBatch{};
    }

    std::vector<PreferenceEncodedSample> encoded;
    std::vector<std::size_t> indices;
    encoded.reserve(samples.size());
    indices.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto item = encode_preference_sample(tokenizer, samples[i], config);
        if (item.chosen_input_ids.empty() || item.rejected_input_ids.empty()) {
            continue;
        }
        if (item.chosen_input_ids == item.rejected_input_ids &&
            item.chosen_response_mask == item.rejected_response_mask) {
            continue;
        }
        encoded.push_back(std::move(item));
        indices.push_back(i);
    }
    return make_preference_batch_from_encoded(encoded, indices, config);
}

JsonlPreferenceDataset::JsonlPreferenceDataset(Tokenizer& tokenizer,
                                               PreferenceBatchConfig config,
                                               std::uint64_t seed)
    : tokenizer_(tokenizer), config_(std::move(config)), rng_(seed) {
    validate_config(config_);
}

void JsonlPreferenceDataset::load(const std::string& path) {
    samples_.clear();
    order_.clear();
    cursor_ = 0;
    kept_samples_ = 0;
    skipped_samples_ = 0;
    identical_after_truncation_ = 0;

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("JsonlPreferenceDataset: cannot open " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        PreferenceSample sample;
        if (!parse_preference_jsonl_row(line, sample)) {
            ++skipped_samples_;
            continue;
        }
        auto encoded = encode_preference_sample(tokenizer_, sample, config_);
        if (encoded.chosen_input_ids.empty() || encoded.rejected_input_ids.empty()) {
            ++skipped_samples_;
            continue;
        }
        if (encoded.chosen_input_ids == encoded.rejected_input_ids &&
            encoded.chosen_response_mask == encoded.rejected_response_mask) {
            ++skipped_samples_;
            ++identical_after_truncation_;
            continue;
        }
        samples_.push_back(std::move(encoded));
        order_.push_back(order_.size());
        ++kept_samples_;
    }
}

void JsonlPreferenceDataset::shuffle() {
    std::shuffle(order_.begin(), order_.end(), rng_);
    cursor_ = 0;
}

PreferenceBatch JsonlPreferenceDataset::next_batch(std::size_t batch_size, bool loop) {
    if (batch_size == 0) {
        throw std::invalid_argument("JsonlPreferenceDataset::next_batch requires batch_size > 0");
    }
    if (samples_.empty()) {
        return PreferenceBatch{};
    }
    if (cursor_ >= samples_.size()) {
        if (!loop) {
            return PreferenceBatch{};
        }
        cursor_ = 0;
    }

    std::vector<PreferenceEncodedSample> rows;
    std::vector<std::size_t> indices;
    rows.reserve(batch_size);
    indices.reserve(batch_size);
    while (rows.size() < batch_size) {
        if (cursor_ >= samples_.size()) {
            if (!loop) {
                break;
            }
            cursor_ = 0;
        }
        const std::size_t ordered_idx = order_[cursor_++];
        rows.push_back(samples_[ordered_idx]);
        indices.push_back(ordered_idx);
    }
    return make_preference_batch_from_encoded(rows, indices, config_);
}

void JsonlPreferenceDataset::reset() {
    cursor_ = 0;
}

std::size_t JsonlPreferenceDataset::num_pairs() const {
    return samples_.size();
}

}  // namespace ops
