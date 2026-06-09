# Android SDK / AAR Packaging

MobileFineTuner ships an Android Library module at:

```text
android-visualizer/mft-sdk
```

The module builds an AAR containing:

- `com.mobilefinetuner.sdk.MobileFineTuner`: Java API for Android apps.
- `libmobilefinetuner_jni.so`: JNI bridge into the native C++ core.
- The compiled MF operator/autograd/model/trainer implementation.

The AAR does not bundle pretrained weights, datasets, adapters, or run logs.
Applications provide HuggingFace-style model directories at runtime, matching
the desktop C++ asset contract in `docs/MODEL_ASSETS.md`.

## Build

Requirements:

- JDK 17+
- Android SDK API 35
- Android NDK `26.1.10909125`
- CMake `3.22.1`

Build the release AAR:

```bash
bash scripts/android/build_mft_sdk_aar.sh
```

Output:

```text
android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar
```

You can also build from the Android project root:

```bash
cd android-visualizer
./gradlew :mft-sdk:assembleRelease
```

Publish to a local Maven repository:

```bash
bash scripts/android/publish_mft_sdk_local.sh
```

Output Maven repository:

```text
android-visualizer/mft-sdk/build/repo
```

Gradle dependency:

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

Override the published version when needed:

```bash
MFT_SDK_VERSION=0.2.0 bash scripts/android/publish_mft_sdk_local.sh
```

## Supported ABI

The SDK currently packages `arm64-v8a` only. This matches the phone-training
target and avoids shipping unvalidated ABI variants.

## Runtime Asset Layout

Push or download model assets into app-readable storage, for example:

```text
/sdcard/Android/data/<your.app.id>/files/models/Qwen2.5-0.5B/
  config.json
  tokenizer.json / tokenizer.model / vocab files
  model.safetensors
  or model.safetensors.index.json + shard files
```

The model directory must be readable by the app process. For production apps,
copy user-selected files into app-private storage before opening the model.

## Java API

```java
import com.mobilefinetuner.sdk.MobileFineTuner;

MobileFineTuner.SelfTestResult smoke = MobileFineTuner.selfTest(context.getFilesDir());

try (MobileFineTuner mf = MobileFineTuner.open(modelDir, true)) {
    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
    mf.createTrainer(MobileFineTuner.TrainerConfig.defaults());

    MobileFineTuner.TrainStepResult result = mf.trainStep(
            inputIds,
            attentionMask,
            labels,
            batchSize,
            sequenceLength
    );

    float loss = result.loss;
}
```

For the standard causal-LM path, Android callers can let the native tokenizer
and batch builder create `inputIds`, `attentionMask`, and shifted labels:

```java
try (MobileFineTuner mf = MobileFineTuner.open(modelDir, true)) {
    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
    mf.createTrainer(MobileFineTuner.TrainerConfig.defaults());

    MobileFineTuner.TrainStepResult result = mf.trainTextBatch(
            new String[]{"A training sentence.", "Another sentence."},
            64,
            true  // appendEos: supervise independent sample boundaries
    );
}
```

Call order:

1. `MobileFineTuner.open(modelDir, loadWeights)`
2. `initLora(...)`
3. `createTrainer(...)`
4. `trainStep(...)` or `trainTextBatch(...)`
5. `close()`

`inputIds`, `attentionMask`, and `labels` are flattened row-major arrays with
length `batchSize * sequenceLength`. Labels use the same shifted causal-LM
contract as the C++ `AutoTrainer`: ignored positions should be `-100`.
`trainTextBatch` applies that contract internally through the C++ tokenizer
factory and `CausalLMBatch` builder. The two-argument `trainTextBatch(texts,
sequenceLength)` keeps `appendEos=false` for backwards compatibility; the
three-argument overload makes the sample-boundary policy explicit.

`TrainerConfig.defaults()` uses streaming full-token CE by default, so Android
training does not allocate a dense `[batch, sequence, vocab]` logits tensor.
Use `new TrainerConfig(lr, wd, maxGradNorm, ignoreIndex, false)` only for dense
logits debugging.

## Device Smoke Test

The SDK includes an instrumentation smoke test that loads the native library and
runs a tiny synthetic GPT-2 LoRA training step on the device:

```bash
cd android-visualizer
./gradlew :mft-sdk:assembleAndroidTest
./gradlew :mft-sdk:connectedDebugAndroidTest
```

The smoke test does not require external model weights. It creates a small
temporary HuggingFace-style config in app-private storage, initializes LoRA, and
executes one native `AutoTrainer` step.

## Sample App

Build the standalone SDK sample app:

```bash
cd android-visualizer
./gradlew :sdk-sample:assembleDebug
```

The sample depends on `:mft-sdk`, displays JNI build information, and runs the
same native self-test on launch.

Run the sample-based phone smoke test:

```bash
bash scripts/android/run_mft_sdk_device_smoke.sh
```

If installation times out, unlock the phone and allow USB app installation, then
rerun the script. Some Android builds route adb installs through a confirmation
screen before package commit.

If the sample app is already installed and you only want to rerun the launch
and native self-test check:

```bash
MFT_SDK_SMOKE_SKIP_INSTALL=1 bash scripts/android/run_mft_sdk_device_smoke.sh
```

## Native Boundary

The JNI bridge is intentionally thin:

```text
Java MobileFineTuner
        |
        v
libmobilefinetuner_jni.so
        |
        v
ops::AutoModelForCausalLM + ops::AutoTrainer
```

The Android layer does not duplicate model math. All forward, loss, backward,
gradient clipping, Adam update, and zero-grad behavior comes from the native C++
core.

## Current Limitations

- `arm64-v8a` only.
- Java API currently exposes the install self-test, one-step tensor LoRA
  training, and one-step text-batch LoRA training, not a full dataset loop or
  checkpoint manager.
- Model files are loaded from normal filesystem paths; the SDK does not load
  directly from compressed APK assets.
- Full-weight phone training still depends on device memory limits and the
  model asset dtype/shape.

These are packaging boundaries, not algorithmic shortcuts. The training step is
the same native MF step used by desktop and adb-driven Android experiments.
