#include "model_family_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace ops {
namespace {

bool contains_module(const std::vector<std::string>& modules, const std::string& name) {
    return std::find(modules.begin(), modules.end(), name) != modules.end();
}

}  // namespace

LoraSpec make_gpt2_lora_spec(const FamilyLoraRequest& request) {
    LoraSpec spec;
    spec.rank = request.rank;
    spec.alpha = request.alpha;
    spec.dropout = request.dropout;
    spec.split_qkv = false;

    const auto& modules = request.target_modules;
    if (modules.empty()) {
        spec.targets = {LoraTarget::AttnQKV, LoraTarget::AttnProj};
        return spec;
    }

    spec.targets.clear();
    const bool wants_fused_qkv =
        contains_module(modules, "attn_qkv") ||
        contains_module(modules, "c_attn") ||
        contains_module(modules, "attn.c_attn");
    const bool wants_split_qkv =
        contains_module(modules, "q_proj") ||
        contains_module(modules, "k_proj") ||
        contains_module(modules, "v_proj");
    if (wants_fused_qkv && wants_split_qkv) {
        throw std::runtime_error(
            "GPT-2 LoRA targets are ambiguous; use either fused "
            "c_attn/attn_qkv or split q_proj/k_proj/v_proj");
    }
    spec.split_qkv = wants_split_qkv;
    if (wants_fused_qkv || wants_split_qkv) {
        spec.targets.push_back(LoraTarget::AttnQKV);
    }
    if (contains_module(modules, "attn_proj") ||
        contains_module(modules, "attn.c_proj") ||
        contains_module(modules, "c_proj") ||
        contains_module(modules, "o_proj")) {
        spec.targets.push_back(LoraTarget::AttnProj);
    }
    if (contains_module(modules, "mlp_fc_in") ||
        contains_module(modules, "mlp.c_fc") ||
        contains_module(modules, "c_fc")) {
        spec.targets.push_back(LoraTarget::MlpFcIn);
    }
    if (contains_module(modules, "mlp_fc_out") ||
        contains_module(modules, "mlp.c_proj") ||
        contains_module(modules, "mlp_c_proj") ||
        contains_module(modules, "c_proj")) {
        spec.targets.push_back(LoraTarget::MlpFcOut);
    }
    if (spec.targets.empty()) {
        throw std::runtime_error("no supported GPT-2 LoRA targets requested");
    }
    return spec;
}

GemmaLoraSpec make_gemma_lora_spec(const FamilyLoraRequest& request) {
    GemmaLoraSpec spec;
    spec.rank = request.rank;
    spec.alpha = request.alpha;
    spec.dropout = request.dropout;
    if (!request.target_modules.empty()) {
        spec.target_modules = request.target_modules;
    }
    return spec;
}

bool is_qv_only_attention_request(const FamilyLoraRequest& request) {
    if (request.target_modules.empty()) {
        return false;
    }
    std::unordered_set<std::string> requested(request.target_modules.begin(),
                                             request.target_modules.end());
    return requested.count("q_proj") > 0 &&
           requested.count("v_proj") > 0 &&
           requested.count("k_proj") == 0 &&
           requested.count("o_proj") == 0;
}

SafeTensorsLoadOptions load_options_for_family(ModelFamily family,
                                               const SafeTensorsLoadOptions& base,
                                               bool verbose) {
    SafeTensorsLoadOptions opts = base;
    opts.verbose = verbose;
    switch (family) {
        case ModelFamily::GPT2:
            // GPT-2 safetensors are already stored in MF's internal [in, out] layout.
            opts.transpose_linear = false;
            break;
        case ModelFamily::Gemma:
        case ModelFamily::Qwen:
        case ModelFamily::Llama:
        case ModelFamily::Mistral:
            // HF decoder linear weights are [out, in]; MF kernels expect [in, out].
            opts.transpose_linear = true;
            break;
        case ModelFamily::Unknown:
            break;
    }
    return opts;
}

}  // namespace ops
