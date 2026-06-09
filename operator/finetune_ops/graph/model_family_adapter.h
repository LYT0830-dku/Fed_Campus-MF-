#pragma once

#include "gemma_lora_injector.h"
#include "lora_injector.h"
#include "model_registry.h"
#include "safetensors_loader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ops {

struct FamilyLoraRequest {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.05f;
    uint64_t seed = 42;
    std::vector<std::string> target_modules;
};

LoraSpec make_gpt2_lora_spec(const FamilyLoraRequest& request);
GemmaLoraSpec make_gemma_lora_spec(const FamilyLoraRequest& request);
bool is_qv_only_attention_request(const FamilyLoraRequest& request);

SafeTensorsLoadOptions load_options_for_family(ModelFamily family,
                                               const SafeTensorsLoadOptions& base,
                                               bool verbose);

}  // namespace ops
