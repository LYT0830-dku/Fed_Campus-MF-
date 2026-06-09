This example provides a minimal runnable workflow for Gemma 3 270M LoRA
fine-tuning and MMLU evaluation.

## Requirements

- C++17, CMake, and a native or Android NDK build environment.
- Python is used only for offline data preparation. Native LoRA training does
  not require PyTorch, Transformers, or PEFT at runtime.
- Dataset paths are resolved from `MFT_DATA_ROOT`, `MMLU_DATA_DIR`,
  `WT2_DATA_DIR`, or the repository-local `data/` fallback.
- Pretrained weights are resolved from `MFT_MODEL_ROOT`,
  `GEMMA_270M_MODEL_DIR`, or the local `pretrained/` fallback.

## Workflow

1. Prepare JSONL training data with masked labels:

   ```bash
   bash run_prepare_data.sh
   ```

   Output: `runs/mmlu_jsonl_gemma270m_s128/` under the repository root.

2. Run LoRA fine-tuning:

   ```bash
   bash run_train.sh
   ```

   Output: `outputs/`.

3. Run MMLU multiple-choice evaluation:

   ```bash
   bash run_eval.sh
   FEWSHOT=5 SPLIT=test bash run_eval.sh
   ```

## Common Overrides

- `run_train.sh`: `TRAIN_MODE`, `MMLU_JSONL`, `WT2_DATA_DIR`, `STEPS`,
  `BATCH_SIZE`, `SEQ_LEN`, `LR`, `RANK`, `ALPHA`, `EPOCHS`, `LOG_EVERY`,
  `GRAD_ACCUM`.
- `run_eval.sh`: `SPLIT`, `FEWSHOT`.
