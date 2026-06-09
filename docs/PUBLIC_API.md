# MobileFineTuner Public API

MobileFineTuner exposes one supported C++ entrypoint:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Downstream projects should link the installed CMake target:

```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(your_target PRIVATE MobileFineTuner::operators)
```

The `finetune_ops/*` headers are still installed because the current examples
use them directly, but new applications should include the umbrella header
first. Historical or non-buildable APIs are excluded from the supported product
surface.

Model weights and datasets are external assets. The library accepts standard
HuggingFace-style model directories with config/tokenizer files and either a
single `model.safetensors` file or `model.safetensors.index.json` plus all
referenced shard files. See `docs/MODEL_ASSETS.md` for the asset contract.

## Stable Auto APIs

Applications should discover model and tokenizer assets through the public
registry APIs instead of hard-coding per-model training directories. For common
fine-tuning code, use `AutoModelForCausalLM` and `AutoTrainer` so the application
does not branch manually between supported GPT-2, Gemma, Qwen, and validated
experimental Llama graph classes.

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);

auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
const auto& load_report = model->safetensors_load_report();
model->init_lora(ops::AutoLoraConfig::attention_qkvo());
auto named_lora_params = model->named_trainable_parameters();

ops::AutoTrainerConfig trainer_cfg;
trainer_cfg.learning_rate = 2e-4f;
ops::AutoTrainer trainer(*model, trainer_cfg);

ops::CausalLMBatchConfig batch_cfg;
batch_cfg.sequence_length = 64;
batch_cfg.append_eos = true;  // optional: supervise independent sample boundaries
auto batch = ops::make_causal_lm_batch(*tokenizer, {"A training sentence."}, batch_cfg);
auto result = trainer.train_step(batch);
```

For multi-step local training, feed batches through the `BatchProvider` surface:

```cpp
ops::WT2Config data_cfg;
data_cfg.train_path = wt2_dir + "/wiki.train.raw";
data_cfg.valid_path = wt2_dir + "/wiki.valid.raw";
data_cfg.seq_len = 64;

ops::WikiText2Dataset dataset(
    data_cfg,
    [&](const std::string& text) {
        auto ids = tokenizer->encode(text);
        return std::vector<int32_t>(ids.begin(), ids.end());
    });
dataset.load(ops::Split::Train);

ops::WikiText2BatchProvider provider(dataset);
ops::AutoFitConfig fit_cfg;
fit_cfg.max_steps = 100;
fit_cfg.batch_size = 4;
auto fit = trainer.fit(provider, fit_cfg);
```

`ModelRegistry` is the C++ equivalent of a small `AutoConfig` layer: it reads
`config.json`, identifies the supported model family, records tokenizer and
SafeTensors asset paths, and exposes default LoRA target names.
`ModelFamilyAdapter` is the policy boundary for family-specific SafeTensors
layout and LoRA target interpretation. `TokenizerFactory` is the matching
`AutoTokenizer` layer: it keeps GPT-2, Qwen, and Gemma tokenizers plus the Llama
3.x adapter behind one `ops::Tokenizer` interface while still using the
model-specific tokenization algorithm required by each checkpoint.
Recognized-only families such as Mistral fail explicitly until their tokenizer
and graph gates pass.

`AutoModelForCausalLM` is the matching model dispatcher. It constructs the
concrete graph class, loads SafeTensors with family-correct transpose defaults,
exposes one `forward(input_ids, attention_mask)` call, initializes LoRA, and
returns trainable parameters. `AutoTrainer` provides the shared LoRA training
core: forward, full-token shifted LM cross entropy, backward, gradient clipping,
Adam update, and zero-grad. Use `train_step(...)` for one-step control or
`fit(BatchProvider&, AutoFitConfig, callback)` for a maintained multi-step loop.
By default it computes the same dense causal-LM objective through
`streaming_lm_cross_entropy`, avoiding materializing the
`[batch, sequence, vocab]` logits tensor on mobile devices. Set
`AutoTrainerConfig::use_streaming_lm_loss = false` only when debugging against a
dense-logits reference path.

Use `named_trainable_parameters()` when exporting adapters, debugging PEFT
alignment, or comparing optimizer-step fixtures. The names are stable internal
module paths such as `layers.0.self_attn.q_proj.lora_A.default.weight`, so tests
do not depend on vector order alone.

After `from_pretrained(..., load_weights=true)`, call
`model->safetensors_load_report()` to inspect which HuggingFace keys were
requested, which tensors were loaded, which keys were missing in non-strict
loads, which checkpoint tensors were not consumed, and which shard file each
loaded tensor came from. This is the supported diagnostic path when onboarding a
new model snapshot or debugging a weight-mapping failure.

For generic causal-LM fine-tuning, use `make_causal_lm_batch` instead of
hand-rolling labels. It tokenizes text, pads/truncates to a fixed sequence
length, builds the attention mask, and creates full-token shifted labels with
padding positions set to `ignore_index`. This keeps the C++ path aligned with
the standard PyTorch/HuggingFace objective:

```text
logits[:, :-1, :] vs labels[:, 1:]
```

`CausalLMBatchConfig::append_eos` controls whether each independent text row
gets one explicit EOS token before truncation and padding. Tokenizers do not
silently append EOS for GPT-2/Qwen; the training data policy lives in the batch
builder.

Prepared task JSONL files use the same tensor contract:

```json
{"ids":[...], "mask":[...], "attention_mask":[...]}
```

`attention_mask` marks real tokens. `mask` is only a label-selection mask for
answer-only objectives such as multiple-choice or QNLI answer-token training.
When a JSONL dataset is loaded with full-token labels enabled, `mask` is ignored
and every real shifted token from `attention_mask` is supervised. Padding
positions remain `ignore_index`.

The current Auto APIs cover the shared causal-LM/LoRA training core. Specialized
dataset loops, logging, checkpoint schedules, and model-specific diagnostics can
still use the lower-level graph classes directly.

## Android SDK API

Android applications can consume the AAR built from `android-visualizer/mft-sdk`.
The Java entrypoint is:

```java
import com.mobilefinetuner.sdk.MobileFineTuner;
```

It wraps the same C++ `AutoModelForCausalLM` and `AutoTrainer` objects through
JNI. The Android SDK is a packaging layer, not a separate implementation. See
`docs/ANDROID_SDK.md` for build, ABI, asset layout, and Java call-order details.

The Android API also exposes `MobileFineTuner.selfTest(File workingDir)`, which
runs a tiny synthetic native LoRA training step on device. This is intended for
installation validation and CI/device smoke testing; real applications should
still pass their own HuggingFace-style `modelDir` to `MobileFineTuner.open`.
