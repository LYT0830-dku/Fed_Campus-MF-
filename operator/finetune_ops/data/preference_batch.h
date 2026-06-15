#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ops {

struct PreferenceBatchConfig {
    int sequence_length = 128;
    int ignore_index = -100;
    bool truncate = true;
    bool add_special_tokens = false;
    bool append_eos_to_response = false;
};

struct PreferenceSample {
    std::string prompt;
    std::string chosen;
    std::string rejected;
    bool has_reference_logps = false;
    float ref_chosen_logp = 0.0f;
    float ref_rejected_logp = 0.0f;
};

struct PreferenceEncodedSample {
    std::vector<int32_t> chosen_input_ids;
    std::vector<int32_t> chosen_attention_mask;
    std::vector<int32_t> chosen_response_mask;

    std::vector<int32_t> rejected_input_ids;
    std::vector<int32_t> rejected_attention_mask;
    std::vector<int32_t> rejected_response_mask;

    bool has_reference_logps = false;
    float ref_chosen_logp = 0.0f;
    float ref_rejected_logp = 0.0f;
};

struct PreferenceBatch {
    TensorPtr chosen_input_ids;       // int32 [B,S]
    TensorPtr chosen_attention_mask;  // float32 [B,S]
    TensorPtr chosen_response_mask;   // int32 [B,S]

    TensorPtr rejected_input_ids;       // int32 [B,S]
    TensorPtr rejected_attention_mask;  // float32 [B,S]
    TensorPtr rejected_response_mask;   // int32 [B,S]

    std::vector<float> ref_chosen_logps;
    std::vector<float> ref_rejected_logps;
    std::vector<std::size_t> sample_indices;

    int batch_size = 0;
    int sequence_length = 0;
    int valid_response_token_count = 0;

    bool has_cached_reference_logps() const;
};

class PreferenceBatchProvider {
public:
    virtual ~PreferenceBatchProvider() = default;

    virtual PreferenceBatch next_batch(std::size_t batch_size, bool loop) = 0;
    virtual void reset() = 0;
    virtual std::size_t num_pairs() const = 0;
};

PreferenceEncodedSample encode_preference_sample(Tokenizer& tokenizer,
                                                 const PreferenceSample& sample,
                                                 const PreferenceBatchConfig& config);

PreferenceBatch make_preference_batch_from_encoded(
    const std::vector<PreferenceEncodedSample>& samples,
    const std::vector<std::size_t>& sample_indices,
    const PreferenceBatchConfig& config);

PreferenceBatch make_preference_batch(Tokenizer& tokenizer,
                                      const std::vector<PreferenceSample>& samples,
                                      const PreferenceBatchConfig& config = PreferenceBatchConfig());

class JsonlPreferenceDataset final : public PreferenceBatchProvider {
public:
    JsonlPreferenceDataset(Tokenizer& tokenizer,
                           PreferenceBatchConfig config = PreferenceBatchConfig(),
                           std::uint64_t seed = 2025);

    void load(const std::string& path);
    void shuffle();

    PreferenceBatch next_batch(std::size_t batch_size, bool loop) override;
    void reset() override;
    std::size_t num_pairs() const override;

    std::size_t num_kept() const { return kept_samples_; }
    std::size_t num_skipped() const { return skipped_samples_; }
    std::size_t num_identical_after_truncation() const { return identical_after_truncation_; }

private:
    Tokenizer& tokenizer_;
    PreferenceBatchConfig config_;
    std::mt19937_64 rng_;
    std::vector<PreferenceEncodedSample> samples_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ = 0;
    std::size_t kept_samples_ = 0;
    std::size_t skipped_samples_ = 0;
    std::size_t identical_after_truncation_ = 0;
};

}  // namespace ops
