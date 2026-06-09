#include "causal_lm_batch.h"
#include "../core/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_eq(int actual, int expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

fs::path make_case_dir() {
    const fs::path root = fs::temp_directory_path() / "mft_causal_lm_batch";
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

void write_minimal_gpt2_assets(const fs::path& root) {
    write_file(root / "config.json", R"({"model_type":"gpt2","vocab_size":50257})");
    write_file(root / "vocab.json", R"({"H":0,"e":1,"<|endoftext|>":50256})");
    write_file(root / "merges.txt", "#version: 0.2\n");
}

class LeftPadTokenizer final : public ops::Tokenizer {
public:
    std::vector<int> encode(const std::string&) override { return {5, 6}; }
    std::string decode(const std::vector<int>&) override { return ""; }
    int get_vocab_size() const override { return 10; }
    int get_eos_token() const override { return 0; }
    int get_bos_token() const override { return 0; }
    int get_pad_token() const override { return 0; }
    int get_unk_token() const override { return 0; }
    bool default_left_padding() const override { return true; }
};

void test_tokenizer_to_full_token_batch() {
    const auto root = make_case_dir();
    write_minimal_gpt2_assets(root);
    auto tokenizer = ops::TokenizerFactory::from_pretrained(root.string());

    ops::CausalLMBatchConfig cfg;
    cfg.sequence_length = 4;
    auto batch = ops::make_causal_lm_batch(*tokenizer, {"He"}, cfg);

    require_eq(batch.batch_size, 1, "batch size mismatch");
    require_eq(batch.sequence_length, 4, "sequence length mismatch");
    require_eq(batch.valid_label_count, 1, "valid label count mismatch");

    const int32_t* ids = batch.input_ids->data<int32_t>();
    const float* mask = batch.attention_mask->data<float>();
    const int32_t* labels = batch.labels->data<int32_t>();

    require_eq(ids[0], 0, "first input token mismatch");
    require_eq(ids[1], 1, "second input token mismatch");
    require_eq(ids[2], 50256, "pad input token mismatch");
    require(mask[0] == 1.0f && mask[1] == 1.0f && mask[2] == 0.0f && mask[3] == 0.0f,
            "attention mask mismatch");
    require_eq(labels[0], -100, "first label must be ignored");
    require_eq(labels[1], 1, "second label should supervise next token");
    require_eq(labels[2], -100, "padding label must be ignored");
}

void test_left_padding_does_not_supervise_from_pad() {
    LeftPadTokenizer tokenizer;
    ops::CausalLMBatchConfig cfg;
    cfg.sequence_length = 4;
    auto batch = ops::make_causal_lm_batch(tokenizer, {"ignored"}, cfg);

    require_eq(batch.valid_label_count, 1, "left-pad valid label count mismatch");

    const int32_t* ids = batch.input_ids->data<int32_t>();
    const float* mask = batch.attention_mask->data<float>();
    const int32_t* labels = batch.labels->data<int32_t>();

    require_eq(ids[0], 0, "left-pad first pad mismatch");
    require_eq(ids[1], 0, "left-pad second pad mismatch");
    require_eq(ids[2], 5, "left-pad first real token mismatch");
    require_eq(ids[3], 6, "left-pad second real token mismatch");
    require(mask[0] == 0.0f && mask[1] == 0.0f && mask[2] == 1.0f && mask[3] == 1.0f,
            "left-pad attention mask mismatch");
    require_eq(labels[0], -100, "left-pad first pad label ignored");
    require_eq(labels[1], -100, "left-pad second pad label ignored");
    require_eq(labels[2], -100, "first real token after pad must be ignored");
    require_eq(labels[3], 6, "second real token should be supervised");
}

void test_append_eos_before_padding() {
    const auto root = make_case_dir();
    write_minimal_gpt2_assets(root);
    auto tokenizer = ops::TokenizerFactory::from_pretrained(root.string());

    ops::CausalLMBatchConfig cfg;
    cfg.sequence_length = 4;
    cfg.append_eos = true;
    auto batch = ops::make_causal_lm_batch(*tokenizer, {"He"}, cfg);

    require_eq(batch.valid_label_count, 2, "append-eos valid label count mismatch");

    const int32_t* ids = batch.input_ids->data<int32_t>();
    const float* mask = batch.attention_mask->data<float>();
    const int32_t* labels = batch.labels->data<int32_t>();

    require_eq(ids[0], 0, "append-eos first token mismatch");
    require_eq(ids[1], 1, "append-eos second token mismatch");
    require_eq(ids[2], 50256, "append-eos real eos token mismatch");
    require_eq(ids[3], 50256, "append-eos pad token mismatch");
    require(mask[0] == 1.0f && mask[1] == 1.0f && mask[2] == 1.0f && mask[3] == 0.0f,
            "append-eos attention mask mismatch");
    require_eq(labels[0], -100, "append-eos first label ignored");
    require_eq(labels[1], 1, "append-eos shifted token label mismatch");
    require_eq(labels[2], 50256, "append-eos shifted eos label mismatch");
    require_eq(labels[3], -100, "append-eos pad label ignored");
}

void test_pretokenized_batch() {
    ops::CausalLMBatchConfig cfg;
    cfg.sequence_length = 5;
    auto batch = ops::make_causal_lm_batch_from_token_ids({{4, 5, 6}, {7, 8}}, 0, cfg);
    require_eq(batch.batch_size, 2, "pretokenized batch size mismatch");
    require_eq(batch.valid_label_count, 3, "pretokenized valid label count mismatch");

    const int32_t* labels = batch.labels->data<int32_t>();
    require_eq(labels[0], -100, "row0 first label ignored");
    require_eq(labels[1], 5, "row0 shifted label 1");
    require_eq(labels[2], 6, "row0 shifted label 2");
    require_eq(labels[5], -100, "row1 first label ignored");
    require_eq(labels[6], 8, "row1 shifted label 1");
    require_eq(labels[7], -100, "row1 pad ignored");
}

}  // namespace

int main() {
    test_tokenizer_to_full_token_batch();
    test_left_padding_does_not_supervise_from_pad();
    test_append_eos_before_padding();
    test_pretokenized_batch();
    return 0;
}
