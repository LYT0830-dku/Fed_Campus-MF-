#include "auto_model.h"
#include "../core/lm_loss.h"
#include "../core/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct AlignmentFixture {
    std::string model_dir;
    std::string model_type;
    std::string prompt;
    std::vector<std::string> prompts;
    int max_length = 0;
    int batch_size = 1;
    int sequence_length = 0;
    int ignore_index = -100;
    float abs_tol = 5e-3f;
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
    std::vector<int> labels;
    int valid_shifted_label_count = 0;
    float expected_loss = 0.0f;
    int last_logits_batch_index = 0;
    int last_logits_position = 0;
    std::vector<int> last_logits_topk_ids;
    std::vector<float> last_logits_topk_values;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open alignment fixture: " + path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

size_t value_start_for_key(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key in alignment fixture: " + key);
    }
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("missing colon for key in alignment fixture: " + key);
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    return pos;
}

bool has_key(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

std::string parse_json_string_at(const std::string& json, size_t quote_pos) {
    require(quote_pos < json.size() && json[quote_pos] == '"', "invalid JSON string start");
    std::string out;
    for (size_t i = quote_pos + 1; i < json.size(); ++i) {
        char c = json[i];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            require(i + 1 < json.size(), "unterminated JSON escape");
            char escaped = json[++i];
            switch (escaped) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    throw std::runtime_error("unsupported JSON escape in alignment fixture");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string in alignment fixture");
}

std::string string_field(const std::string& json, const std::string& key, const std::string& fallback = "") {
    if (!has_key(json, key)) {
        return fallback;
    }
    return parse_json_string_at(json, value_start_for_key(json, key));
}

std::vector<std::string> string_array_field(const std::string& json, const std::string& key) {
    size_t pos = value_start_for_key(json, key);
    require(pos < json.size() && json[pos] == '[', "expected string array for key: " + key);
    ++pos;
    std::vector<std::string> values;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
            ++pos;
        }
        if (pos < json.size() && json[pos] == ']') {
            return values;
        }
        require(pos < json.size() && json[pos] == '"', "expected string in array for key: " + key);
        values.push_back(parse_json_string_at(json, pos));
        ++pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != ']') {
            ++pos;
        }
    }
    throw std::runtime_error("unterminated string array for key: " + key);
}

int int_field(const std::string& json, const std::string& key, int fallback = 0) {
    if (!has_key(json, key)) {
        return fallback;
    }
    size_t pos = value_start_for_key(json, key);
    const size_t start = pos;
    if (pos < json.size() && json[pos] == '-') {
        ++pos;
    }
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    require(start != pos, "expected integer for key: " + key);
    return std::stoi(json.substr(start, pos - start));
}

float float_field(const std::string& json, const std::string& key, float fallback = 0.0f) {
    if (!has_key(json, key)) {
        return fallback;
    }
    size_t pos = value_start_for_key(json, key);
    const size_t start = pos;
    if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
        ++pos;
    }
    while (pos < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[pos])) ||
            json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' ||
            json[pos] == '-' || json[pos] == '+')) {
        ++pos;
    }
    require(start != pos, "expected float for key: " + key);
    return std::stof(json.substr(start, pos - start));
}

std::vector<int> int_array_field(const std::string& json, const std::string& key) {
    size_t pos = value_start_for_key(json, key);
    require(pos < json.size() && json[pos] == '[', "expected int array for key: " + key);
    ++pos;
    std::vector<int> values;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
            ++pos;
        }
        if (pos < json.size() && json[pos] == ']') {
            return values;
        }
        const size_t start = pos;
        if (pos < json.size() && json[pos] == '-') {
            ++pos;
        }
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        require(start != pos, "expected integer in array for key: " + key);
        values.push_back(std::stoi(json.substr(start, pos - start)));
    }
    throw std::runtime_error("unterminated int array for key: " + key);
}

std::vector<float> float_array_field(const std::string& json, const std::string& key) {
    size_t pos = value_start_for_key(json, key);
    require(pos < json.size() && json[pos] == '[', "expected float array for key: " + key);
    ++pos;
    std::vector<float> values;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
            ++pos;
        }
        if (pos < json.size() && json[pos] == ']') {
            return values;
        }
        const size_t start = pos;
        if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
            ++pos;
        }
        while (pos < json.size() &&
               (std::isdigit(static_cast<unsigned char>(json[pos])) ||
                json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' ||
                json[pos] == '-' || json[pos] == '+')) {
            ++pos;
        }
        require(start != pos, "expected float in array for key: " + key);
        values.push_back(std::stof(json.substr(start, pos - start)));
    }
    throw std::runtime_error("unterminated float array for key: " + key);
}

AlignmentFixture load_fixture(const std::string& path) {
    const std::string json = read_file(path);
    AlignmentFixture fixture;
    fixture.model_dir = string_field(json, "model_dir");
    fixture.model_type = string_field(json, "model_type");
    fixture.prompt = string_field(json, "prompt");
    if (has_key(json, "prompts")) {
        fixture.prompts = string_array_field(json, "prompts");
    } else if (!fixture.prompt.empty()) {
        fixture.prompts = {fixture.prompt};
    }
    fixture.max_length = int_field(json, "max_length", 0);
    fixture.batch_size = int_field(json, "batch_size", 1);
    fixture.sequence_length = int_field(json, "sequence_length", 0);
    fixture.ignore_index = int_field(json, "ignore_index", -100);
    fixture.abs_tol = float_field(json, "abs_tol", 5e-3f);
    fixture.input_ids = int_array_field(json, "input_ids");
    fixture.attention_mask = int_array_field(json, "attention_mask");
    fixture.labels = int_array_field(json, "labels");
    fixture.valid_shifted_label_count = int_field(json, "valid_shifted_label_count", 0);
    fixture.expected_loss = float_field(json, "expected_loss", 0.0f);
    fixture.last_logits_batch_index = int_field(json, "last_logits_batch_index", 0);
    fixture.last_logits_position = int_field(json, "last_logits_position", 0);
    fixture.last_logits_topk_ids = int_array_field(json, "last_logits_topk_ids");
    fixture.last_logits_topk_values = float_array_field(json, "last_logits_topk_values");
    require(!fixture.model_dir.empty(), "fixture model_dir is empty");
    require(!fixture.prompts.empty(), "fixture prompts are empty");
    require(!fixture.input_ids.empty(), "fixture input_ids is empty");
    require(fixture.input_ids.size() == fixture.attention_mask.size(), "input/mask size mismatch");
    require(fixture.input_ids.size() == fixture.labels.size(), "input/labels size mismatch");
    require(fixture.batch_size > 0, "fixture batch_size must be positive");
    if (fixture.sequence_length == 0) {
        require(fixture.batch_size == 1, "fixture sequence_length is required for batch fixtures");
        fixture.sequence_length = static_cast<int>(fixture.input_ids.size());
    }
    require(static_cast<size_t>(fixture.batch_size * fixture.sequence_length) == fixture.input_ids.size(),
            "fixture input_ids size does not match batch_size * sequence_length");
    require(fixture.last_logits_batch_index >= 0 &&
            fixture.last_logits_batch_index < fixture.batch_size,
            "fixture last_logits_batch_index out of range");
    require(fixture.last_logits_topk_ids.size() == fixture.last_logits_topk_values.size(),
            "top-k ids/values size mismatch");
    return fixture;
}

ops::TensorPtr make_int_tensor(const std::vector<int>& values, int batch_size, int sequence_length) {
    std::vector<int32_t> data(values.begin(), values.end());
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{batch_size, sequence_length},
        data.data(),
        ops::kInt32,
        ops::kCPU);
}

ops::TensorPtr make_mask_tensor(const std::vector<int>& values, int batch_size, int sequence_length) {
    std::vector<float> data;
    data.reserve(values.size());
    for (int v : values) {
        data.push_back(static_cast<float>(v));
    }
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{batch_size, sequence_length},
        data.data(),
        ops::kFloat32,
        ops::kCPU);
}

std::vector<int> flatten_encoded_ids(ops::Tokenizer& tokenizer,
                                     const std::vector<std::string>& prompts,
                                     int max_length) {
    std::vector<int> flattened;
    if (max_length > 0) {
        flattened.reserve(prompts.size() * static_cast<size_t>(max_length));
        for (const auto& prompt : prompts) {
            const auto encoded = tokenizer.encode_with_attention(prompt, max_length, true);
            flattened.insert(flattened.end(), encoded.input_ids.begin(), encoded.input_ids.end());
        }
        return flattened;
    }

    require(prompts.size() == 1, "unpadded alignment fixtures must contain one prompt");
    ops::TokenizerEncodeOptions encode_options;
    encode_options.add_special_tokens = true;
    encode_options.truncation = true;
    return tokenizer.encode_with_options(prompts[0], encode_options);
}

std::vector<int> flatten_attention_mask(ops::Tokenizer& tokenizer,
                                        const std::vector<std::string>& prompts,
                                        int max_length) {
    if (max_length <= 0) {
        return std::vector<int>(flatten_encoded_ids(tokenizer, prompts, max_length).size(), 1);
    }

    std::vector<int> flattened;
    flattened.reserve(prompts.size() * static_cast<size_t>(max_length));
    for (const auto& prompt : prompts) {
        const auto encoded = tokenizer.encode_with_attention(prompt, max_length, true);
        flattened.insert(flattened.end(), encoded.attention_mask.begin(), encoded.attention_mask.end());
    }
    return flattened;
}

std::vector<std::pair<int, float>> topk(const float* values, int64_t count, size_t k) {
    std::vector<std::pair<int, float>> scored;
    scored.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        scored.emplace_back(static_cast<int>(i), values[i]);
    }
    const size_t keep = std::min(k, scored.size());
    std::partial_sort(
        scored.begin(),
        scored.begin() + static_cast<std::ptrdiff_t>(keep),
        scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    scored.resize(keep);
    return scored;
}

void compare_vector(const std::vector<int>& actual,
                    const std::vector<int>& expected,
                    const std::string& name) {
    if (actual == expected) {
        return;
    }
    size_t mismatch = 0;
    while (mismatch < actual.size() && mismatch < expected.size() &&
           actual[mismatch] == expected[mismatch]) {
        ++mismatch;
    }
    throw std::runtime_error(
        name + " mismatch at index " + std::to_string(mismatch) +
        " actual_len=" + std::to_string(actual.size()) +
        " expected_len=" + std::to_string(expected.size()));
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture_path;
    if (argc > 1) {
        fixture_path = argv[1];
    } else if (const char* env = std::getenv("MFT_LM_ALIGNMENT_FIXTURE");
               env != nullptr && env[0] != '\0') {
        fixture_path = env;
    } else {
        std::cout << "[SKIP] MFT_LM_ALIGNMENT_FIXTURE is not set; "
                  << "generate one with scripts/generate_lm_alignment_fixture.py\n";
        return 0;
    }

    const AlignmentFixture fixture = load_fixture(fixture_path);

    ops::TokenizerLoadOptions tokenizer_options;
    tokenizer_options.model_type = fixture.model_type;
    auto tokenizer = ops::TokenizerFactory::from_pretrained(fixture.model_dir, tokenizer_options);

    compare_vector(flatten_encoded_ids(*tokenizer, fixture.prompts, fixture.max_length),
                   fixture.input_ids,
                   "tokenizer input_ids");
    compare_vector(flatten_attention_mask(*tokenizer, fixture.prompts, fixture.max_length),
                   fixture.attention_mask,
                   "tokenizer attention_mask");

    ops::AutoModelLoadOptions load_options;
    load_options.verbose = false;
    auto model = ops::AutoModelForCausalLM::from_pretrained(fixture.model_dir, load_options);

    auto input_ids = make_int_tensor(
        fixture.input_ids, fixture.batch_size, fixture.sequence_length);
    auto attention_mask = make_mask_tensor(
        fixture.attention_mask, fixture.batch_size, fixture.sequence_length);
    auto labels = make_int_tensor(
        fixture.labels, fixture.batch_size, fixture.sequence_length);

    auto logits = model->forward(input_ids, attention_mask);
    auto loss = ops::lm_cross_entropy(logits, labels, fixture.ignore_index, "mean");
    const float actual_loss = loss->data<float>()[0];
    const float loss_diff = std::fabs(actual_loss - fixture.expected_loss);
    std::cout << "loss actual=" << actual_loss
              << " expected=" << fixture.expected_loss
              << " diff=" << loss_diff << std::endl;
    require(loss_diff <= fixture.abs_tol,
            "loss mismatch exceeds abs_tol=" + std::to_string(fixture.abs_tol));

    const auto& shape = logits->shape();
    require(shape.size() == 3, "logits must be [B,S,V]");
    require(shape[0] == fixture.batch_size, "logits batch size mismatch");
    const int64_t seq_len = shape[1];
    const int64_t vocab = shape[2];
    require(fixture.last_logits_position >= 0 && fixture.last_logits_position < seq_len,
            "fixture last_logits_position out of range");

    const float* data = logits->data<float>();
    const float* row = data +
        (static_cast<int64_t>(fixture.last_logits_batch_index) * seq_len +
         static_cast<int64_t>(fixture.last_logits_position)) * vocab;
    const auto actual_topk = topk(row, vocab, fixture.last_logits_topk_ids.size());

    for (size_t i = 0; i < actual_topk.size(); ++i) {
        const int actual_id = actual_topk[i].first;
        const float actual_value = actual_topk[i].second;
        const int expected_id = fixture.last_logits_topk_ids[i];
        const float expected_value = fixture.last_logits_topk_values[i];
        const float value_diff = std::fabs(actual_value - expected_value);
        std::cout << "top" << i
                  << " actual=(" << actual_id << "," << actual_value << ")"
                  << " expected=(" << expected_id << "," << expected_value << ")"
                  << " diff=" << value_diff << std::endl;
        require(actual_id == expected_id, "last-token top-k id mismatch");
        require(value_diff <= fixture.abs_tol,
                "last-token top-k value mismatch exceeds abs_tol=" +
                    std::to_string(fixture.abs_tol));
    }

    std::cout << "AutoModel alignment passed: " << fixture_path << std::endl;
    return 0;
}
