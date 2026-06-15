#include "mobile_finetuner/mobile_finetuner.h"

#include "finetune_ops/core/memory_manager.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string model_dir;
    std::string preference_jsonl;
    int seq_len = 64;
    int batch_size = 1;
    int max_steps = 10;
    float lr = 2e-4f;
    float beta = 0.1f;
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.05f;
    uint64_t seed = 42;
    bool shuffle = false;
    bool cached_ref_logps = false;
    bool dense_loss = false;
};

std::string require_value(int& i, int argc, char** argv, const std::string& key) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
    }
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--model_dir") {
            args.model_dir = require_value(i, argc, argv, key);
        } else if (key == "--preference_jsonl") {
            args.preference_jsonl = require_value(i, argc, argv, key);
        } else if (key == "--seq_len") {
            args.seq_len = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--batch_size") {
            args.batch_size = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--max_steps") {
            args.max_steps = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--lr") {
            args.lr = std::stof(require_value(i, argc, argv, key));
        } else if (key == "--beta") {
            args.beta = std::stof(require_value(i, argc, argv, key));
        } else if (key == "--lora_r") {
            args.lora_r = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--lora_alpha") {
            args.lora_alpha = std::stof(require_value(i, argc, argv, key));
        } else if (key == "--lora_dropout") {
            args.lora_dropout = std::stof(require_value(i, argc, argv, key));
        } else if (key == "--seed") {
            args.seed = static_cast<uint64_t>(std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--shuffle") {
            args.shuffle = true;
        } else if (key == "--cached_ref_logps") {
            args.cached_ref_logps = true;
        } else if (key == "--dense_loss") {
            args.dense_loss = true;
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty()) {
        throw std::runtime_error("--model_dir is required");
    }
    if (args.preference_jsonl.empty()) {
        throw std::runtime_error("--preference_jsonl is required");
    }
    if (!std::filesystem::is_directory(args.model_dir)) {
        throw std::runtime_error("model_dir does not exist: " + args.model_dir);
    }
    if (!std::filesystem::is_regular_file(args.preference_jsonl)) {
        throw std::runtime_error("preference_jsonl does not exist: " + args.preference_jsonl);
    }
    if (args.seq_len <= 1 || args.batch_size <= 0 || args.max_steps <= 0) {
        throw std::runtime_error("seq_len, batch_size, and max_steps must be positive");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        std::cout << "========== Qwen DPO LoRA Run ==========\n";
        std::cout << "model_dir=" << args.model_dir << "\n";
        std::cout << "preference_jsonl=" << args.preference_jsonl << "\n";
        std::cout << "batch_size=" << args.batch_size
                  << " seq_len=" << args.seq_len
                  << " max_steps=" << args.max_steps
                  << " lr=" << args.lr
                  << " beta=" << args.beta
                  << " loss_impl=" << (args.dense_loss ? "dense" : "streaming")
                  << "\n";

        auto tokenizer = ops::TokenizerFactory::from_pretrained(args.model_dir);

        ops::PreferenceBatchConfig batch_cfg;
        batch_cfg.sequence_length = args.seq_len;
        batch_cfg.append_eos_to_response = true;

        ops::JsonlPreferenceDataset dataset(*tokenizer, batch_cfg, args.seed);
        dataset.load(args.preference_jsonl);
        if (args.shuffle) {
            dataset.shuffle();
        }
        if (dataset.num_pairs() == 0) {
            throw std::runtime_error("no valid preference pairs after tokenization/truncation");
        }
        std::cout << "loaded_pairs=" << dataset.num_pairs()
                  << " skipped_pairs=" << dataset.num_skipped() << "\n";

        ops::AutoModelLoadOptions load_options;
        load_options.load_weights = true;
        load_options.verbose = false;

        std::cout << "loading policy model...\n";
        auto policy = ops::AutoModelForCausalLM::from_pretrained(args.model_dir, load_options);

        ops::AutoLoraConfig lora;
        lora.rank = args.lora_r;
        lora.alpha = args.lora_alpha;
        lora.dropout = args.lora_dropout;
        lora.seed = args.seed;
        lora.target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"};
        policy->init_lora(lora);

        ops::DPOTrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = args.lr;
        trainer_cfg.beta = args.beta;
        trainer_cfg.max_grad_norm = 1.0f;
        trainer_cfg.use_streaming_dpo_loss = !args.dense_loss;

        std::unique_ptr<ops::AutoModelForCausalLM> reference;
        std::unique_ptr<ops::DPOTrainer> trainer;
        if (args.cached_ref_logps) {
            trainer = std::make_unique<ops::DPOTrainer>(*policy, trainer_cfg);
        } else {
            std::cout << "loading frozen reference model...\n";
            reference = ops::AutoModelForCausalLM::from_pretrained(args.model_dir, load_options);
            trainer = std::make_unique<ops::DPOTrainer>(*policy, *reference, trainer_cfg);
        }

        double total_step_seconds = 0.0;
        for (int step = 1; step <= args.max_steps; ++step) {
            auto batch = dataset.next_batch(static_cast<std::size_t>(args.batch_size), true);
            const auto t0 = std::chrono::steady_clock::now();
            const ops::DPOTrainStepResult result = trainer->train_step(batch);
            const auto t1 = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(t1 - t0).count();
            total_step_seconds += seconds;

            std::cout << std::fixed << std::setprecision(4)
                      << "[step " << step << "/" << args.max_steps << "] "
                      << "loss=" << result.loss
                      << " margin=" << result.reward_margin
                      << " acc=" << result.reward_accuracy
                      << " response_tokens=" << result.valid_response_token_count
                      << " time_s=" << seconds
                      << "\n";
            ops::MemoryManager::instance().force_cleanup();
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "mean_step_time_s=" << (total_step_seconds / args.max_steps)
                  << " total_train_time_s=" << total_step_seconds << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
