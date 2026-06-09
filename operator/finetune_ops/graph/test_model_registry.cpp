#include "model_registry.h"
#include "model_family_adapter.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

fs::path make_case_dir(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("mft_model_registry_" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

void test_qwen_registry() {
    const fs::path root = make_case_dir("qwen");
    write_file(root / "config.json",
               R"({"model_type":"qwen2","tie_word_embeddings":false})");
    write_file(root / "tokenizer.json", "{}");
    write_file(root / "vocab.json", "{}");
    write_file(root / "merges.txt", "#version: 0.2\n");
    write_file(root / "model.safetensors.index.json", "{}");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Qwen, "qwen family inference failed");
    require(spec.model_type == "qwen2", "qwen model_type mismatch");
    require(!spec.tie_word_embeddings, "qwen tie_word_embeddings should be false");
    require(spec.assets.has_tokenizer_json, "qwen tokenizer.json not detected");
    require(spec.assets.has_vocab_json, "qwen vocab.json not detected");
    require(spec.assets.has_merges_txt, "qwen merges.txt not detected");
    require(!spec.assets.has_single_safetensors, "qwen single safetensors should be absent");
    require(spec.assets.has_sharded_safetensors, "qwen safetensors index not detected");
    require(spec.default_lora_targets.size() == 4, "qwen default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "qwen first default LoRA target mismatch");
    require(ops::to_string(spec.family) == "qwen", "qwen family string mismatch");
}

void test_gpt2_registry_defaults() {
    const fs::path root = make_case_dir("gpt2");
    write_file(root / "config.json", R"({"model_type":"gpt2"})");
    write_file(root / "model.safetensors", "");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::GPT2, "gpt2 family inference failed");
    require(spec.tie_word_embeddings, "gpt2 tie_word_embeddings should default true");
    require(spec.assets.has_single_safetensors, "gpt2 single safetensors not detected");
    require(!spec.assets.has_sharded_safetensors, "gpt2 sharded index should be absent");
    require(spec.default_lora_targets.size() == 2, "gpt2 default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "attn_qkv", "gpt2 first LoRA target mismatch");
}

void test_gemma_registry_defaults_match_training_path() {
    const fs::path root = make_case_dir("gemma");
    write_file(root / "config.json", R"({"model_type":"gemma3_text"})");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Gemma, "gemma family inference failed");
    require(spec.default_lora_targets.size() == 7, "gemma default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "gemma first LoRA target mismatch");
    require(spec.default_lora_targets[4] == "gate_proj", "gemma MLP gate target missing");
    require(spec.default_lora_targets[6] == "down_proj", "gemma MLP down target missing");
}

void test_llama_registry_defaults_match_hf_layout() {
    const fs::path root = make_case_dir("llama");
    write_file(root / "config.json",
               R"({"model_type":"llama","tie_word_embeddings":false})");
    write_file(root / "tokenizer.model", "");
    write_file(root / "model.safetensors.index.json", "{}");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Llama, "llama family inference failed");
    require(spec.model_type == "llama", "llama model_type mismatch");
    require(!spec.tie_word_embeddings, "llama tie_word_embeddings should default/load false");
    require(spec.assets.has_sharded_safetensors, "llama safetensors index not detected");
    require(spec.default_lora_targets.size() == 4, "llama default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "llama first default LoRA target mismatch");
    require(spec.default_lora_targets[3] == "o_proj", "llama o_proj target missing");
    require(ops::to_string(spec.family) == "llama", "llama family string mismatch");
}

void test_mistral_registry_recognizes_next_family_without_claiming_graph() {
    const fs::path root = make_case_dir("mistral");
    write_file(root / "config.json",
               R"({"model_type":"mistral","tie_word_embeddings":false})");
    write_file(root / "tokenizer.model", "");
    write_file(root / "model.safetensors.index.json", "{}");

    const auto spec = ops::ModelRegistry::inspect_pretrained(root.string());
    require(spec.family == ops::ModelFamily::Mistral, "mistral family inference failed");
    require(spec.model_type == "mistral", "mistral model_type mismatch");
    require(!spec.tie_word_embeddings, "mistral tie_word_embeddings should default/load false");
    require(spec.assets.has_sharded_safetensors, "mistral safetensors index not detected");
    require(spec.default_lora_targets.size() == 4, "mistral default LoRA target count mismatch");
    require(spec.default_lora_targets[0] == "q_proj", "mistral first default LoRA target mismatch");
    require(spec.default_lora_targets[3] == "o_proj", "mistral o_proj target missing");
    require(ops::to_string(spec.family) == "mistral", "mistral family string mismatch");
}

void test_unknown_model_type() {
    const fs::path root = make_case_dir("unknown");
    write_file(root / "config.json", R"({"model_type":"mamba"})");

    bool threw = false;
    try {
        (void)ops::ModelRegistry::inspect_pretrained(root.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "unknown model_type did not throw");
}

void test_family_adapter_gpt2_lora_policy() {
    ops::FamilyLoraRequest request;
    request.rank = 2;
    request.alpha = 4.0f;
    request.dropout = 0.0f;

    auto default_spec = ops::make_gpt2_lora_spec(request);
    require(!default_spec.split_qkv, "GPT-2 default policy should use fused qkv");
    require(default_spec.targets.size() == 2, "GPT-2 default target count mismatch");
    require(default_spec.targets[0] == ops::LoraTarget::AttnQKV,
            "GPT-2 default must include attention qkv");
    require(default_spec.targets[1] == ops::LoraTarget::AttnProj,
            "GPT-2 default must include attention projection");

    request.target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"};
    auto split_spec = ops::make_gpt2_lora_spec(request);
    require(split_spec.split_qkv, "GPT-2 explicit q/k/v policy should split qkv");
    require(split_spec.targets.size() == 2, "GPT-2 split target count mismatch");

    request.target_modules = {"c_attn", "c_proj"};
    auto peft_spec = ops::make_gpt2_lora_spec(request);
    require(!peft_spec.split_qkv, "GPT-2 PEFT c_attn policy should stay fused");
    require(peft_spec.targets.size() == 3,
            "GPT-2 PEFT c_attn+c_proj should include fused qkv, attn proj, and mlp output");
    require(peft_spec.targets[2] == ops::LoraTarget::MlpFcOut,
            "GPT-2 PEFT c_proj should map to MLP output projection as well");

    request.target_modules = {"c_attn", "q_proj"};
    bool threw = false;
    try {
        (void)ops::make_gpt2_lora_spec(request);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("ambiguous") != std::string::npos;
    }
    require(threw, "GPT-2 mixed fused/split qkv request should be rejected");
}

void test_family_adapter_load_options() {
    ops::SafeTensorsLoadOptions base;
    base.transpose_linear = true;
    base.verbose = true;

    auto gpt2 = ops::load_options_for_family(ops::ModelFamily::GPT2, base, false);
    require(!gpt2.transpose_linear, "GPT-2 load policy should not transpose linear weights");
    require(!gpt2.verbose, "GPT-2 load policy should apply requested verbosity");

    auto qwen = ops::load_options_for_family(ops::ModelFamily::Qwen, base, false);
    require(qwen.transpose_linear, "Qwen load policy should transpose HF linear weights");

    auto llama = ops::load_options_for_family(ops::ModelFamily::Llama, base, true);
    require(llama.transpose_linear, "Llama load policy should transpose HF linear weights");
    require(llama.verbose, "Llama load policy should apply requested verbosity");
}

}  // namespace

int main() {
    test_qwen_registry();
    test_gpt2_registry_defaults();
    test_gemma_registry_defaults_match_training_path();
    test_llama_registry_defaults_match_hf_layout();
    test_mistral_registry_recognizes_next_family_without_claiming_graph();
    test_unknown_model_type();
    test_family_adapter_gpt2_lora_policy();
    test_family_adapter_load_options();
    return 0;
}
