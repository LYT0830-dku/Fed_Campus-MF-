# Android Training App

MobileFineTuner ships a user-facing Android application in:

```text
android-visualizer/app
```

The app is the recommended on-device entry point for users. It combines:

- a foreground SDK training runner;
- local model import into app-private storage;
- native tokenizer-backed training through `mft-sdk`;
- run-log visualization for existing experiment folders.

## Build

```bash
cd android-visualizer
./gradlew :app:assembleDebug
```

Debug APK:

```text
android-visualizer/app/build/outputs/apk/debug/app-debug.apk
```

## Runtime Flow

1. Open the app.
2. Use the `Run` tab.
3. Tap `Import Model`.
4. Select a HuggingFace-style model directory containing `config.json`,
   tokenizer files, and optionally SafeTensors weights.
5. Select the imported local model.
6. Choose whether to load SafeTensors weights.
7. Set sequence length and step count.
8. Start training.

The app copies selected model files into:

```text
/data/user/0/com.mobilefinetuner.visualizer/files/models/<model-name>
```

This avoids Android scoped-storage failures when native C++ code opens model
files through normal filesystem paths.

## Supported Runner Path

The app calls the Android SDK in this order:

1. `MobileFineTuner.open(modelDir, loadWeights)`
2. `initLora(MobileFineTuner.LoraConfig.attentionQkvo())`
3. `createTrainer(...)`
4. `trainTextBatch(...)`

The runner is intended to run while the app is visible. The activity keeps the
screen awake during training so vendor background execution policies are less
likely to stop the process.

## Notes

- `loadWeights=false` validates config, tokenizer, graph construction, LoRA,
  trainer creation, and native training wiring without loading full weights.
- `loadWeights=true` is the full pretrained-weight path and requires enough
  device memory and foreground execution time.
- The existing dashboard tabs remain available for parsing and visualizing
  historical run folders containing logs and RSS traces.
