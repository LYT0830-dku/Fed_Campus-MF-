#include "preference_batch.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

class TinyTokenizer : public ops::Tokenizer {
public:
    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            if (c >= 'a' && c <= 'z') {
                ids.push_back(1 + (c - 'a'));
            } else if (c >= 'A' && c <= 'Z') {
                ids.push_back(1 + (c - 'A'));
            } else if (c == ' ') {
                ids.push_back(30);
            } else {
                ids.push_back(31);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>&) override { return ""; }
    int get_vocab_size() const override { return 64; }
    int get_eos_token() const override { return 0; }
    int get_bos_token() const override { return 0; }
    int get_pad_token() const override { return 0; }
    int get_unk_token() const override { return 31; }
};

class LeftPadTokenizer final : public TinyTokenizer {
public:
    bool default_left_padding() const override { return true; }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_i32(int32_t actual, int32_t expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void require_f32(float actual, float expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void test_make_preference_batch() {
    TinyTokenizer tokenizer;
    ops::PreferenceBatchConfig cfg;
    cfg.sequence_length = 5;
    auto batch = ops::make_preference_batch(
        tokenizer,
        {ops::PreferenceSample{"p", "ab", "cd", true, -1.5f, -2.0f}},
        cfg);

    require(batch.batch_size == 1, "preference batch size mismatch");
    require(batch.sequence_length == 5, "preference sequence length mismatch");
    require(batch.has_cached_reference_logps(), "preference reference logps missing");
    require(batch.valid_response_token_count == 4, "valid response token count mismatch");

    const int32_t* chosen_ids = batch.chosen_input_ids->data<int32_t>();
    const int32_t* chosen_mask = batch.chosen_response_mask->data<int32_t>();
    const float* chosen_attn = batch.chosen_attention_mask->data<float>();

    require_i32(chosen_ids[0], 16, "prompt token mismatch");
    require_i32(chosen_ids[1], 1, "chosen first response token mismatch");
    require_i32(chosen_ids[2], 2, "chosen second response token mismatch");
    require_i32(chosen_ids[3], 0, "pad token mismatch");
    require_i32(chosen_mask[0], 0, "prompt response mask mismatch");
    require_i32(chosen_mask[1], 1, "response mask token 1 mismatch");
    require_i32(chosen_mask[2], 1, "response mask token 2 mismatch");
    require_f32(chosen_attn[3], 0.0f, "padding attention mismatch");
}

void test_left_padding_keeps_response_mask_aligned() {
    LeftPadTokenizer tokenizer;
    ops::PreferenceBatchConfig cfg;
    cfg.sequence_length = 5;
    auto batch = ops::make_preference_batch(
        tokenizer,
        {ops::PreferenceSample{"p", "a", "b"}},
        cfg);

    const int32_t* chosen_ids = batch.chosen_input_ids->data<int32_t>();
    const int32_t* chosen_mask = batch.chosen_response_mask->data<int32_t>();
    const float* chosen_attn = batch.chosen_attention_mask->data<float>();

    require_i32(chosen_ids[0], 0, "left-pad first token mismatch");
    require_i32(chosen_ids[3], 16, "left-pad prompt token mismatch");
    require_i32(chosen_ids[4], 1, "left-pad response token mismatch");
    require_i32(chosen_mask[4], 1, "left-pad response mask mismatch");
    require_f32(chosen_attn[0], 0.0f, "left-pad attention mismatch");
}

void test_jsonl_dataset() {
    const fs::path root = fs::temp_directory_path() / "mft_preference_batch";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path path = root / "prefs.jsonl";
    {
        std::ofstream out(path);
        out << R"({"prompt":"p","chosen":"ab","rejected":"cd","ref_chosen_logp":-1.25,"ref_rejected_logp":-2.5})" << "\n";
        out << R"({"prompt":"x","chosen":"","rejected":"y"})" << "\n";
        out << R"({"bad":true})" << "\n";
    }

    TinyTokenizer tokenizer;
    ops::PreferenceBatchConfig cfg;
    cfg.sequence_length = 5;
    ops::JsonlPreferenceDataset dataset(tokenizer, cfg, 7);
    dataset.load(path.string());

    require(dataset.num_pairs() == 1, "jsonl kept pair count mismatch");
    require(dataset.num_kept() == 1, "jsonl kept metric mismatch");
    require(dataset.num_skipped() == 2, "jsonl skipped metric mismatch");
    auto batch = dataset.next_batch(1, false);
    require(batch.has_cached_reference_logps(), "jsonl cached refs missing");
    require_f32(batch.ref_chosen_logps[0], -1.25f, "jsonl ref chosen mismatch");
    require_f32(batch.ref_rejected_logps[0], -2.5f, "jsonl ref rejected mismatch");

    fs::remove_all(root);
}

}  // namespace

int main() {
    test_make_preference_batch();
    test_left_padding_keeps_response_mask_aligned();
    test_jsonl_dataset();
    return 0;
}
