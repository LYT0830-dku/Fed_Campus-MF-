#include "auto_model.h"

#include "../core/memory_manager.h"
#include "model_family_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ops {

namespace {

FamilyLoraRequest make_family_lora_request(const AutoLoraConfig& cfg) {
    FamilyLoraRequest request;
    request.rank = cfg.rank;
    request.alpha = cfg.alpha;
    request.dropout = cfg.dropout;
    request.seed = cfg.seed;
    request.target_modules = cfg.target_modules;
    return request;
}

}  // namespace

AutoModelForCausalLM::AutoModelForCausalLM(std::string model_dir,
                                           ModelArchitectureSpec spec)
    : model_dir_(std::move(model_dir)), spec_(std::move(spec)) {}

std::unique_ptr<AutoModelForCausalLM>
AutoModelForCausalLM::from_pretrained(const std::string& model_dir,
                                      const AutoModelLoadOptions& options) {
    auto spec = ModelRegistry::inspect_pretrained(model_dir);
    auto model = std::unique_ptr<AutoModelForCausalLM>(
        new AutoModelForCausalLM(model_dir, spec));

    switch (model->spec_.family) {
        case ModelFamily::GPT2: {
            GPT2Config cfg = GPT2Config::from_pretrained(model_dir);
            model->gpt2_ = std::make_unique<GPT2Model>(cfg);
            if (cfg.tie_word_embeddings) {
                model->gpt2_->tie_weights();
            }
            break;
        }
        case ModelFamily::Gemma: {
            GemmaTextConfig cfg = GemmaTextConfig::from_pretrained(model_dir);
            model->gemma_ = std::make_unique<GemmaModel>(cfg);
            break;
        }
        case ModelFamily::Llama: {
            LlamaConfig cfg = LlamaConfig::from_pretrained(model_dir + "/config.json");
            model->llama_ = std::make_unique<LlamaModel>(cfg);
            break;
        }
        case ModelFamily::Mistral:
            throw std::runtime_error(
                "AutoModelForCausalLM: Mistral is recognized by ModelRegistry, "
                "but the Mistral graph/tokenizer gates have not passed yet");
        case ModelFamily::Qwen: {
            if (!model->spec_.tie_word_embeddings) {
                throw std::runtime_error(
                    "AutoModelForCausalLM: Qwen checkpoints with tie_word_embeddings=false "
                    "are not supported yet; provide a tied Qwen checkpoint or add lm_head.weight support");
            }
            QwenConfig cfg = QwenConfig::from_pretrained(model_dir + "/config.json");
            model->qwen_ = std::make_unique<QwenModel>(cfg);
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }

    if (options.load_weights) {
        model->load_pretrained_weights(options);
    }
    return model;
}

void AutoModelForCausalLM::load_pretrained_weights(const AutoModelLoadOptions& options) {
    SafeTensorsModelReader reader(model_dir_);
    reader.parse_headers();
    safetensors_load_report_.clear();

    const auto load_opts =
        load_options_for_family(spec_.family, options.safetensors_options, options.verbose);

    switch (spec_.family) {
        case ModelFamily::GPT2: {
            auto mapping = GPT2KeyMapper::generate_gpt2_mapping(gpt2_->config().n_layer);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                gpt2_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            break;
        }
        case ModelFamily::Gemma: {
            auto mapping = GemmaKeyMapper::generate_gemma_mapping(gemma_->config().num_hidden_layers);
            const auto tensor_names = reader.get_tensor_names();
            const bool has_lm_head =
                std::find(tensor_names.begin(), tensor_names.end(), "lm_head.weight") != tensor_names.end();
            if (!has_lm_head) {
                mapping.erase("lm_head.weight");
            }
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                gemma_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            break;
        }
        case ModelFamily::Llama: {
            auto mapping = LlamaKeyMapper::generate_llama_mapping(
                llama_->config().num_hidden_layers,
                llama_->config().tie_word_embeddings,
                llama_->config().attention_bias);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                llama_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            MemoryManager::instance().clear_unused_memory();
            break;
        }
        case ModelFamily::Qwen: {
            auto mapping = QwenKeyMapper::generate_qwen_mapping(qwen_->config().num_hidden_layers);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                qwen_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            MemoryManager::instance().clear_unused_memory();
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::forward(const TensorPtr& input_ids,
                                        const TensorPtr& attention_mask) {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->forward(input_ids, attention_mask);
        case ModelFamily::Gemma:
            return gemma_->forward(input_ids, attention_mask);
        case ModelFamily::Llama:
            return llama_->forward(input_ids, attention_mask);
        case ModelFamily::Qwen:
            return qwen_->forward(input_ids, attention_mask);
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::forward_hidden(const TensorPtr& input_ids,
                                               const TensorPtr& attention_mask) {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Gemma:
            return gemma_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Llama:
            return llama_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Qwen:
            return qwen_->forward_hidden(input_ids, attention_mask);
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::lm_head_weight_for_loss() const {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->lm_head_weight();
        case ModelFamily::Gemma:
            return gemma_->lm_head_weight_for_loss();
        case ModelFamily::Llama:
            return llama_->lm_head_weight_for_loss();
        case ModelFamily::Qwen:
            return qwen_->embedding_weight();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

void AutoModelForCausalLM::init_lora(const AutoLoraConfig& config) {
    switch (spec_.family) {
        case ModelFamily::GPT2: {
            gpt2_->init_lora_modules();
            gpt2_lora_ = std::make_unique<LoraInjector>();
            gpt2_lora_->inject(*gpt2_, make_gpt2_lora_spec(make_family_lora_request(config)));
            break;
        }
        case ModelFamily::Gemma: {
            gemma_lora_ = std::make_unique<GemmaLoraInjector>();
            gemma_lora_->inject(*gemma_, make_gemma_lora_spec(make_family_lora_request(config)));
            break;
        }
        case ModelFamily::Llama:
            llama_->init_lora(config.rank, config.alpha, config.dropout,
                              is_qv_only_attention_request(make_family_lora_request(config)),
                              config.seed);
            llama_->freeze_base();
            break;
        case ModelFamily::Qwen:
            qwen_->init_lora(config.rank, config.alpha, config.dropout,
                             is_qv_only_attention_request(make_family_lora_request(config)),
                             config.seed);
            qwen_->freeze_base();
            break;
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->parameters();
        case ModelFamily::Gemma:
            return gemma_->parameters();
        case ModelFamily::Llama:
            return llama_->parameters();
        case ModelFamily::Qwen:
            return qwen_->parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::trainable_parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->get_lora_parameters();
        case ModelFamily::Gemma:
            return gemma_->get_lora_parameters();
        case ModelFamily::Llama:
            return llama_->get_lora_parameters();
        case ModelFamily::Qwen:
            return qwen_->get_lora_parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<std::pair<std::string, TensorPtr>> AutoModelForCausalLM::named_trainable_parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->named_lora_parameters();
        case ModelFamily::Gemma:
            return gemma_->named_lora_parameters();
        case ModelFamily::Llama:
            return llama_->named_lora_parameters();
        case ModelFamily::Qwen:
            return qwen_->named_lora_parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

}  // namespace ops
