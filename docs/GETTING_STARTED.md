# Getting Started

This guide describes the supported consumer workflow for MobileFineTuner as a
local C++ and Android SDK product. MobileFineTuner does not ship model weights
or datasets; applications provide HuggingFace-style assets at runtime.

## 1. Prepare Model Assets

Create one model root and place local snapshots under it:

```bash
export MFT_MODEL_ROOT=/path/to/mft-models

huggingface-cli download gpt2 \
  --local-dir "$MFT_MODEL_ROOT/gpt2" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir "$MFT_MODEL_ROOT/Qwen2.5-0.5B" \
  --local-dir-use-symlinks False
```

Each model directory must contain `config.json`, tokenizer assets, and either
`model.safetensors` or `model.safetensors.index.json` plus its shard files.
See `docs/MODEL_ASSETS.md` for the full contract.

## 2. Use the C++ API

Include the stable umbrella header:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Minimal causal-LM LoRA step:

```cpp
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);

auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);
model->init_lora(ops::AutoLoraConfig::attention_qkvo());

ops::AutoTrainerConfig trainer_cfg;
trainer_cfg.learning_rate = 2e-4f;
ops::AutoTrainer trainer(*model, trainer_cfg);

ops::CausalLMBatchConfig batch_cfg;
batch_cfg.sequence_length = 64;
batch_cfg.append_eos = true;

auto batch = ops::make_causal_lm_batch(
    *tokenizer,
    {"Question: ... Answer: ..."},
    batch_cfg
);

auto result = trainer.train_step(batch);
```

Recommended CMake consumer setup:

```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(your_target PRIVATE MobileFineTuner::operators)
```

## 3. Use the Android SDK

Build the AAR:

```bash
bash scripts/android/build_mft_sdk_aar.sh
```

The AAR is written to:

```text
android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar
```

Publish to a local Maven repository:

```bash
bash scripts/android/publish_mft_sdk_local.sh
```

Gradle consumer dependency:

```kotlin
repositories {
    maven {
        url = uri("/path/to/MobileFineTuner/android-visualizer/mft-sdk/build/repo")
    }
}

dependencies {
    implementation("com.mobilefinetuner:mobilefinetuner-android:0.1.0")
}
```

Minimal Java call flow:

```java
try (MobileFineTuner mf = MobileFineTuner.open(modelDir, true)) {
    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
    mf.createTrainer(MobileFineTuner.TrainerConfig.defaults());
    MobileFineTuner.TrainStepResult result = mf.trainTextBatch(
            new String[]{"A training sentence."},
            64,
            true
    );
}
```

See `docs/ANDROID_SDK.md` for ABI, asset placement, smoke tests, and call-order
details.

## 4. Add a New Model Family

Do not add a new family by copying an example directory. Add it through the
model layer:

1. Extend `ModelRegistry` to recognize the family from `config.json`.
2. Add or select the tokenizer implementation in `TokenizerFactory`.
3. Add the graph implementation or adapter in `operator/finetune_ops/graph`.
4. Add load-layout and LoRA target policy in `ModelFamilyAdapter`.
5. Add tokenizer golden alignment fixtures against HuggingFace.
6. Add forward/loss alignment and PEFT one-step LoRA optimizer-step fixtures.
7. Run the Android smoke gate when the family is intended for phone training.

This keeps the C++ path aligned with the PyTorch/HuggingFace/PEFT reference
instead of relying on model-specific example scripts.

## 5. Local Release Gates

Before handing a local build to another user, run:

```bash
ctest --test-dir /tmp/mft-lm-alignment --output-on-failure
bash scripts/check_release_tree.sh
git diff --check
bash scripts/android/build_mft_sdk_aar.sh
```

For phone validation:

```bash
bash scripts/android/run_mft_sdk_device_smoke.sh
```

The release-tree audit checks that archived experiments are isolated, package
files exist, shell entrypoints parse, no large source-tree assets leaked in, and
no CJK text remains in maintained release files.
