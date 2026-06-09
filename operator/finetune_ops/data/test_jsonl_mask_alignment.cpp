#include "wikitext2_dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ops;

namespace {

void require_eq_i32(int32_t actual, int32_t expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void require_eq_f32(float actual, float expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void test_answer_mask_jsonl() {
    namespace fs = std::filesystem;

    const fs::path tmp_dir = fs::temp_directory_path() / "mf_jsonl_mask_alignment";
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);
    const fs::path jsonl_path = tmp_dir / "train.jsonl";

    {
        std::ofstream out(jsonl_path);
        out << R"({"ids":[10,11,12],"mask":[0,0,1]})" << "\n";
        out << R"({"ids":[20,21,22,23,24],"mask":[0,1,0,1,1],"attention_mask":[1,1,1,0,0]})" << "\n";
    }

    WT2Config cfg;
    cfg.jsonl_train = jsonl_path.string();
    cfg.seq_len = 5;
    cfg.pad_id = 99;
    cfg.shuffle_train = false;

    WikiText2Dataset ds(cfg, [](const std::string&) { return std::vector<int32_t>{}; });
    ds.load(Split::Train);
    Batch batch = ds.next_batch(2, false);

    const int32_t* input_ids = batch.input_ids->data<int32_t>();
    const int32_t* labels = batch.labels->data<int32_t>();
    const float* attention_mask = batch.attention_mask->data<float>();

    const int32_t expected_input_ids[10] = {
        10, 11, 12, 99, 99,
        20, 21, 22, 23, 24,
    };
    const int32_t expected_labels[10] = {
        -100, -100, 12, -100, -100,
        -100, 21, -100, -100, -100,
    };
    const float expected_attention[10] = {
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    };

    for (int i = 0; i < 10; ++i) {
        require_eq_i32(input_ids[i], expected_input_ids[i],
                       "answer-mask input_ids mismatch at index " + std::to_string(i));
        require_eq_i32(labels[i], expected_labels[i],
                       "answer-mask labels mismatch at index " + std::to_string(i));
        require_eq_f32(attention_mask[i], expected_attention[i],
                       "answer-mask attention_mask mismatch at index " + std::to_string(i));
    }

    fs::remove_all(tmp_dir);
}

void test_full_token_jsonl_respects_attention_mask() {
    namespace fs = std::filesystem;

    const fs::path tmp_dir = fs::temp_directory_path() / "mf_jsonl_full_token_alignment";
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);
    const fs::path jsonl_path = tmp_dir / "train.jsonl";

    {
        std::ofstream out(jsonl_path);
        out << R"({"ids":[30,31,32,99,99],"mask":[0,0,1,0,0],"attention_mask":[1,1,1,0,0]})" << "\n";
        out << R"({"ids":[40,41,42,43,99],"mask":[0,0,0,1,0],"attention_mask":[1,1,1,1,0]})" << "\n";
    }

    WT2Config cfg;
    cfg.jsonl_train = jsonl_path.string();
    cfg.seq_len = 5;
    cfg.pad_id = 99;
    cfg.shuffle_train = false;
    cfg.jsonl_full_token_labels = true;

    WikiText2Dataset ds(cfg, [](const std::string&) { return std::vector<int32_t>{}; });
    ds.load(Split::Train);
    Batch batch = ds.next_batch(2, false);

    const int32_t* labels = batch.labels->data<int32_t>();
    const float* attention_mask = batch.attention_mask->data<float>();

    const int32_t expected_labels[10] = {
        -100, 31, 32, -100, -100,
        -100, 41, 42, 43, -100,
    };
    const float expected_attention[10] = {
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    };

    for (int i = 0; i < 10; ++i) {
        require_eq_i32(labels[i], expected_labels[i],
                       "full-token labels mismatch at index " + std::to_string(i));
        require_eq_f32(attention_mask[i], expected_attention[i],
                       "full-token attention_mask mismatch at index " + std::to_string(i));
    }

    fs::remove_all(tmp_dir);
}

}  // namespace

int main() {
    test_answer_mask_jsonl();
    test_full_token_jsonl_respects_attention_mask();

    std::cout << "JSONL mask alignment test passed\n";
    return 0;
}
