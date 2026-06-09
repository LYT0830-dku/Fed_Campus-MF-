#include "auto_model.h"
#include "../optim/auto_trainer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct TensorRecord {
    std::string name;
    std::vector<float> values;
};

struct PeftStepFixture {
    std::string schema;
    std::string model_dir;
    std::string model_type;
    std::string mf_lora_layout;
    int batch_size = 0;
    int sequence_length = 0;
    int ignore_index = -100;
    int rank = 0;
    int seed = 42;
    int trainable_tensor_count = 0;
    float alpha = 0.0f;
    float dropout = 0.0f;
    float learning_rate = 0.0f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    float expected_loss = 0.0f;
    float abs_tol = 5e-3f;
    int valid_shifted_label_count = 0;
    bool base_model_dropout_disabled = true;
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
    std::vector<int> labels;
    std::vector<std::string> target_modules;
    std::vector<TensorRecord> initial_lora_tensors;
    std::vector<TensorRecord> expected_after_step_lora_tensors;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open PEFT fixture: " + path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

bool has_key(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

size_t value_start_for_key(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key in PEFT fixture: " + key);
    }
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("missing colon for key in PEFT fixture: " + key);
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    return pos;
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
                    throw std::runtime_error("unsupported JSON escape in PEFT fixture");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string in PEFT fixture");
}

std::string string_field(const std::string& json, const std::string& key, const std::string& fallback = "") {
    if (!has_key(json, key)) {
        return fallback;
    }
    return parse_json_string_at(json, value_start_for_key(json, key));
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

bool bool_field(const std::string& json, const std::string& key, bool fallback = false) {
    if (!has_key(json, key)) {
        return fallback;
    }
    const size_t pos = value_start_for_key(json, key);
    if (json.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        return false;
    }
    throw std::runtime_error("expected boolean for key: " + key);
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

size_t find_matching(const std::string& json, size_t open_pos, char open, char close) {
    require(open_pos < json.size() && json[open_pos] == open, "invalid JSON container start");
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = open_pos; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open) {
            ++depth;
        } else if (c == close) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    throw std::runtime_error("unterminated JSON container");
}

std::vector<std::string> object_array_field(const std::string& json, const std::string& key) {
    size_t pos = value_start_for_key(json, key);
    require(pos < json.size() && json[pos] == '[', "expected object array for key: " + key);
    const size_t end = find_matching(json, pos, '[', ']');
    ++pos;
    std::vector<std::string> objects;
    while (pos < end) {
        while (pos < end &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) {
            ++pos;
        }
        if (pos >= end) {
            break;
        }
        require(json[pos] == '{', "expected object in array for key: " + key);
        const size_t obj_end = find_matching(json, pos, '{', '}');
        objects.push_back(json.substr(pos, obj_end - pos + 1));
        pos = obj_end + 1;
    }
    return objects;
}

std::vector<TensorRecord> tensor_records_field(const std::string& json, const std::string& key) {
    std::vector<TensorRecord> records;
    for (const auto& object : object_array_field(json, key)) {
        TensorRecord record;
        record.name = string_field(object, "name");
        record.values = float_array_field(object, "values");
        require(!record.name.empty(), "tensor record has empty name");
        records.push_back(std::move(record));
    }
    return records;
}

void require_tensor_records_well_formed(const std::vector<TensorRecord>& records,
                                        const std::string& field_name) {
    std::unordered_set<std::string> seen;
    for (const auto& record : records) {
        require(!record.name.empty(), field_name + " contains an empty tensor name");
        require(seen.insert(record.name).second,
                field_name + " contains a duplicate tensor name: " + record.name);
        require(!record.values.empty(), field_name + " contains an empty tensor: " + record.name);
        for (float value : record.values) {
            require(std::isfinite(value),
                    field_name + " contains NaN/Inf tensor value: " + record.name);
        }
    }
}

void require_tensor_record_sets_match(const std::vector<TensorRecord>& initial,
                                      const std::vector<TensorRecord>& after) {
    std::unordered_map<std::string, size_t> initial_sizes;
    for (const auto& record : initial) {
        initial_sizes.emplace(record.name, record.values.size());
    }
    for (const auto& record : after) {
        const auto it = initial_sizes.find(record.name);
        require(it != initial_sizes.end(),
                "after-step tensor missing from initial set: " + record.name);
        require(it->second == record.values.size(),
                "initial/after tensor value count mismatch for: " + record.name);
    }
}

PeftStepFixture load_fixture(const std::string& path) {
    const std::string json = read_file(path);
    PeftStepFixture fixture;
    fixture.schema = string_field(json, "schema");
    fixture.model_dir = string_field(json, "model_dir");
    fixture.model_type = string_field(json, "model_type");
    fixture.mf_lora_layout = string_field(json, "mf_lora_layout");
    fixture.batch_size = int_field(json, "batch_size");
    fixture.sequence_length = int_field(json, "sequence_length");
    fixture.ignore_index = int_field(json, "ignore_index", -100);
    fixture.rank = int_field(json, "rank");
    fixture.seed = int_field(json, "seed", 42);
    fixture.trainable_tensor_count = int_field(json, "trainable_tensor_count", 0);
    fixture.alpha = float_field(json, "alpha");
    fixture.dropout = float_field(json, "dropout", 0.0f);
    fixture.learning_rate = float_field(json, "learning_rate");
    fixture.weight_decay = float_field(json, "weight_decay", 0.0f);
    fixture.max_grad_norm = float_field(json, "max_grad_norm", 1.0f);
    fixture.expected_loss = float_field(json, "expected_loss");
    fixture.abs_tol = float_field(json, "abs_tol", 5e-3f);
    fixture.valid_shifted_label_count = int_field(json, "valid_shifted_label_count", 0);
    fixture.base_model_dropout_disabled =
        bool_field(json, "base_model_dropout_disabled", true);
    fixture.input_ids = int_array_field(json, "input_ids");
    fixture.attention_mask = int_array_field(json, "attention_mask");
    fixture.labels = int_array_field(json, "labels");
    fixture.target_modules = string_array_field(json, "target_modules");
    fixture.initial_lora_tensors = tensor_records_field(json, "initial_lora_tensors");
    fixture.expected_after_step_lora_tensors =
        tensor_records_field(json, "expected_after_step_lora_tensors");

    require(fixture.schema == "mobilefinetuner.peft_lora_step.v1",
            "unsupported PEFT fixture schema: " + fixture.schema);
    require(!fixture.model_dir.empty(), "fixture model_dir is empty");
    require(!fixture.model_type.empty(), "fixture model_type is empty");
    require(fixture.mf_lora_layout == "peft" || fixture.mf_lora_layout == "in_rank",
            "fixture mf_lora_layout must be peft or in_rank");
    require(fixture.batch_size > 0, "fixture batch_size must be positive");
    require(fixture.sequence_length > 1, "fixture sequence_length must be greater than one");
    require(fixture.rank > 0, "fixture rank must be positive");
    require(fixture.alpha > 0.0f, "fixture alpha must be positive");
    require(fixture.dropout >= 0.0f && fixture.dropout < 1.0f,
            "fixture dropout must be in [0, 1)");
    require(fixture.learning_rate > 0.0f, "fixture learning_rate must be positive");
    require(fixture.weight_decay >= 0.0f, "fixture weight_decay must be non-negative");
    require(fixture.max_grad_norm >= 0.0f, "fixture max_grad_norm must be non-negative");
    require(fixture.abs_tol > 0.0f, "fixture abs_tol must be positive");
    require(fixture.valid_shifted_label_count > 0,
            "fixture valid_shifted_label_count must be positive");
    require(!fixture.target_modules.empty(), "fixture target_modules is empty");
    const size_t expected_tokens = static_cast<size_t>(fixture.batch_size) *
                                   static_cast<size_t>(fixture.sequence_length);
    require(fixture.input_ids.size() == expected_tokens, "fixture input_ids size mismatch");
    require(fixture.attention_mask.size() == expected_tokens, "fixture attention_mask size mismatch");
    require(fixture.labels.size() == expected_tokens, "fixture labels size mismatch");
    require(!fixture.initial_lora_tensors.empty(), "fixture initial LoRA tensors are empty");
    require(fixture.expected_after_step_lora_tensors.size() == fixture.initial_lora_tensors.size(),
            "fixture initial/after LoRA tensor count mismatch");
    require(fixture.trainable_tensor_count ==
                static_cast<int>(fixture.initial_lora_tensors.size()),
            "fixture trainable_tensor_count does not match LoRA tensor records");
    require_tensor_records_well_formed(fixture.initial_lora_tensors,
                                       "initial_lora_tensors");
    require_tensor_records_well_formed(fixture.expected_after_step_lora_tensors,
                                       "expected_after_step_lora_tensors");
    require_tensor_record_sets_match(fixture.initial_lora_tensors,
                                     fixture.expected_after_step_lora_tensors);
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
    for (int value : values) {
        data.push_back(static_cast<float>(value));
    }
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{batch_size, sequence_length},
        data.data(),
        ops::kFloat32,
        ops::kCPU);
}

std::unordered_map<std::string, ops::TensorPtr>
named_map(const std::vector<std::pair<std::string, ops::TensorPtr>>& named) {
    std::unordered_map<std::string, ops::TensorPtr> result;
    for (const auto& kv : named) {
        require(result.emplace(kv.first, kv.second).second,
                "duplicate MF named trainable parameter: " + kv.first);
    }
    return result;
}

void copy_values_to_tensor(const TensorRecord& record, const ops::TensorPtr& tensor) {
    require(tensor != nullptr, "null MF tensor for record: " + record.name);
    require(tensor->dtype() == ops::kFloat32, "MF LoRA tensor is not float32: " + record.name);
    require(tensor->numel() == static_cast<int64_t>(record.values.size()),
            "MF LoRA tensor numel mismatch for " + record.name +
                ": actual=" + std::to_string(tensor->numel()) +
                " fixture=" + std::to_string(record.values.size()));
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = record.values[static_cast<size_t>(i)];
    }
}

float max_abs_diff(const ops::TensorPtr& tensor, const std::vector<float>& expected) {
    require(tensor != nullptr, "null tensor in max_abs_diff");
    require(tensor->numel() == static_cast<int64_t>(expected.size()),
            "tensor numel mismatch in max_abs_diff");
    const float* data = tensor->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        max_diff = std::max(max_diff,
                            std::fabs(data[i] - expected[static_cast<size_t>(i)]));
    }
    return max_diff;
}

void require_record_names_match(const std::vector<TensorRecord>& expected,
                                const std::unordered_map<std::string, ops::TensorPtr>& actual,
                                const std::string& phase) {
    for (const auto& record : expected) {
        if (actual.find(record.name) == actual.end()) {
            std::ostringstream oss;
            oss << "PEFT fixture " << phase << " tensor name not found in MF: "
                << record.name << "\nMF names:";
            for (const auto& kv : actual) {
                oss << "\n  " << kv.first;
            }
            throw std::runtime_error(oss.str());
        }
    }
}

}  // namespace

int main() {
    const char* fixture_env = std::getenv("MFT_PEFT_LORA_STEP_FIXTURE");
    if (!fixture_env || std::string(fixture_env).empty()) {
        std::cout << "[SKIP] MFT_PEFT_LORA_STEP_FIXTURE is not set" << std::endl;
        return 0;
    }

    const auto fixture = load_fixture(fixture_env);

    ops::AutoModelLoadOptions load_options;
    load_options.verbose = false;
    auto model = ops::AutoModelForCausalLM::from_pretrained(fixture.model_dir, load_options);

    ops::AutoLoraConfig lora_config;
    lora_config.rank = fixture.rank;
    lora_config.alpha = fixture.alpha;
    lora_config.dropout = fixture.dropout;
    lora_config.seed = static_cast<uint64_t>(fixture.seed);
    lora_config.target_modules = fixture.target_modules;
    model->init_lora(lora_config);

    auto named_params = named_map(model->named_trainable_parameters());
    require_record_names_match(fixture.initial_lora_tensors, named_params, "initial");
    require_record_names_match(fixture.expected_after_step_lora_tensors, named_params, "after-step");
    require(named_params.size() == fixture.initial_lora_tensors.size(),
            "MF named trainable count differs from PEFT fixture count");

    for (const auto& record : fixture.initial_lora_tensors) {
        copy_values_to_tensor(record, named_params.at(record.name));
    }

    ops::AutoTrainerConfig trainer_config;
    trainer_config.learning_rate = fixture.learning_rate;
    trainer_config.weight_decay = fixture.weight_decay;
    trainer_config.max_grad_norm = fixture.max_grad_norm;
    trainer_config.ignore_index = fixture.ignore_index;
    trainer_config.use_streaming_lm_loss = true;

    ops::AutoTrainer trainer(*model, trainer_config);
    auto input_ids = make_int_tensor(fixture.input_ids, fixture.batch_size, fixture.sequence_length);
    auto attention_mask = make_mask_tensor(fixture.attention_mask,
                                           fixture.batch_size,
                                           fixture.sequence_length);
    auto labels = make_int_tensor(fixture.labels, fixture.batch_size, fixture.sequence_length);

    const auto result = trainer.train_step(input_ids, attention_mask, labels);
    require(result.valid_label_count == fixture.valid_shifted_label_count,
            "valid shifted label count mismatch");

    const float loss_diff = std::fabs(result.loss - fixture.expected_loss);
    if (loss_diff > fixture.abs_tol) {
        throw std::runtime_error("PEFT step loss mismatch: actual=" +
                                 std::to_string(result.loss) +
                                 " expected=" + std::to_string(fixture.expected_loss) +
                                 " diff=" + std::to_string(loss_diff) +
                                 " tolerance=" + std::to_string(fixture.abs_tol));
    }

    for (const auto& record : fixture.expected_after_step_lora_tensors) {
        const float diff = max_abs_diff(named_params.at(record.name), record.values);
        if (diff > fixture.abs_tol) {
            throw std::runtime_error("PEFT after-step LoRA tensor mismatch for " +
                                     record.name + ": max_abs_diff=" +
                                     std::to_string(diff) +
                                     " tolerance=" + std::to_string(fixture.abs_tol));
        }
    }

    std::cout << "[PASS] PEFT LoRA one-step alignment fixture"
              << " model_type=" << fixture.model_type
              << " loss=" << result.loss
              << " trainable_tensors=" << result.trainable_tensor_count
              << std::endl;
    return 0;
}
