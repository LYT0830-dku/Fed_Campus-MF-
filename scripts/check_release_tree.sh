#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

fail=0

say() {
  printf '%s\n' "$*"
}

check_no_personal_paths() {
  local files=(
	    README.md
	    docs/ANDROID_SDK.md
	    docs/CURRENT_FAMILY_TASK_MATRIX.md
	    docs/MODEL_ASSETS.md
    scripts/README.md
    scripts/asset_paths.py
    scripts/lib/asset_paths.sh
    scripts/check_local_assets.sh
	    scripts/check_tokenizer_hf_alignment.sh
	    scripts/prepare_arcc_jsonl.py
	    scripts/prepare_mcq_jsonl.py
	    scripts/prepare_qnli_jsonl.py
	    scripts/check_peft_lora_step_fixture.sh
    scripts/run_training_smoke.sh
    scripts/run_training_real_assets.sh
    scripts/android/android_env.sh
    scripts/android/adb_resource_monitor.sh
    scripts/android/build_mft_sdk_aar.sh
    scripts/android/build_qwen_android.sh
    scripts/android/publish_mft_sdk_local.sh
    scripts/android/run_mft_sdk_device_smoke.sh
    scripts/android/run_qwen_qnli_native_phone.sh
    scripts/android/stage_qwen_qnli_phone_assets.sh
    operator/CMakeLists.txt
    operator/cmake/OperatorsConfig.cmake.in
    operator/cmake/MobileFineTunerConfig.cmake.in
    operator/include/mobile_finetuner/mobile_finetuner.h
  )
  local matches
  matches="$(cd "$ROOT" && rg -n '/Users/|Documents/pretrained_models|Documents/MobileFineTuner|Documents/MobileFinetuner' "${files[@]}" || true)"
  if [ -n "$matches" ]; then
    say "[FAIL] Personal paths found in maintained release files:"
    say "$matches"
    fail=1
  else
    say "[OK]   No personal absolute paths in maintained release files"
  fi
}

check_no_archived_experiment_surface() {
  local matches
  matches="$(
    cd "$ROOT" && rg -n 'ONNX|Onnx|onnx|\bORT\b|OrtTraining|Termux|termux' \
      README.md docs scripts android-visualizer/app/src/main android-visualizer/app/build.gradle.kts operator .github \
      --glob '!Rubbish/**' \
      --glob '!scripts/check_release_tree.sh' \
      --glob '!operator/build*/**' \
      --glob '!**/build/**' \
      --glob '!**/build-android/**' \
      --glob '!review-stage/**' \
      || true
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Archived ONNX/Termux experiment surface leaked outside Rubbish:"
    say "$matches"
    fail=1
  else
    say "[OK]   ONNX/Termux experiment surface is isolated in Rubbish"
  fi

  matches="$(
    cd "$ROOT" && find . \
      -path './.git' -prune -o \
      -path './.agents' -prune -o \
      -path './.aris' -prune -o \
      -path './Rubbish' -prune -o \
      -path './operator/build*' -prune -o \
      -path './*/build' -prune -o \
      -path './*/build-android' -prune -o \
      \( -path './operator/opt_ops' \
         -o -path './pytorch_alignment' \
         -o -path './scripts/Finetune' \
         -o -path './scripts/onnx_lora' \
         -o -path './operator/finetune_ops/graph/test_safetensors_simple' \
         -o -path './operator/finetune_ops/graph/pt_last_logits.json' \
         -o -path './operator/finetune_ops/graph/layer0_activations.json' \
         -o -path './operator/finetune_ops/graph/save_pt_gold.py' \) \
      -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Archived experimental/generated files remain in the release tree:"
    say "$matches"
    fail=1
  else
    say "[OK]   Experimental/generated source artifacts are isolated in Rubbish"
  fi
}

check_large_source_files() {
  local matches
  matches="$(
    cd "$ROOT" && find . -type f -size +50M \
      -not -path './.git/*' \
      -not -path './.agents/*' \
      -not -path './.aris/*' \
      -not -path './.venv*/*' \
      -not -path './data/*' \
      -not -path './runs/*' \
      -not -path './pytorch_runs/*' \
      -not -path './review-stage/*' \
      -not -path './Rubbish/*' \
      -not -path './android-visualizer/*/.cxx/*' \
      -not -path './android-visualizer/*/build/*' \
      -not -path './operator/build*/*' \
      -not -path './*/build/*' \
      -not -path './*/build-android/*' \
      -not -path './*/pretrained/*' \
      -not -path './*/outputs/*' \
      -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Large files outside ignored asset/build areas:"
    say "$matches"
    fail=1
  else
    say "[OK]   No >50MB source-tree files outside ignored asset/build areas"
  fi
}

check_package_files() {
  local required=(
    operator/cmake/OperatorsConfig.cmake.in
    operator/cmake/MobileFineTunerConfig.cmake.in
    operator/include/mobile_finetuner/mobile_finetuner.h
    docs/GETTING_STARTED.md
    docs/MODEL_ASSETS.md
    docs/PUBLIC_API.md
  )
  local path
  local missing=0
  for path in "${required[@]}"; do
    if [ ! -f "$ROOT/$path" ]; then
      say "[FAIL] Missing package/documentation file: $path"
      fail=1
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    say "[OK]   Package config templates and asset docs exist"
  fi
}

check_no_cjk_text_in_maintained_tree() {
  local matches
  matches="$(
    cd "$ROOT" && python3 - <<'PY'
import os
from pathlib import Path

roots = [
    Path("README.md"),
    Path("docs"),
    Path("scripts"),
    Path("android-visualizer/mft-sdk"),
    Path("android-visualizer/sdk-sample"),
    Path("operator"),
    Path("examples"),
]
prune_names = {".git", ".gradle", ".cxx", "build", "build-android", "Rubbish", "runs", "review-stage"}
binary_suffixes = {".zip", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".a", ".o", ".so"}

def has_han(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0x30000 <= code <= 0x3134F
        ):
            return True
    return False

def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in prune_names]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in binary_suffixes:
                continue
            yield path

for root in roots:
    for path in iter_files(root):
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(data.splitlines(), 1):
            if has_han(line):
                print(f"{path}:{line_no}:{line}")
PY
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] CJK text found in maintained release files:"
    say "$matches"
    fail=1
  else
    say "[OK]   No CJK text in maintained release files"
  fi
}

check_no_generated_caches() {
  local matches
  matches="$(
    cd "$ROOT" && find . \
      -path './.git' -prune -o \
      -path './.agents' -prune -o \
      -path './.aris' -prune -o \
      -path './Rubbish' -prune -o \
      -path './runs' -prune -o \
      -path './review-stage' -prune -o \
      -name __pycache__ -type d -print
  )"
  if [ -n "$matches" ]; then
    say "[FAIL] Python cache directories remain in the maintained tree:"
    say "$matches"
    fail=1
  else
    say "[OK]   No Python cache directories in maintained tree"
  fi
}

check_no_stale_operator_api() {
  local stale=(
    operator/finetune_ops/model
    operator/finetune_ops/nn/embedding.cpp
    operator/finetune_ops/nn/embedding.h
    operator/finetune_ops/optim/lora_optimizer.cpp
    operator/finetune_ops/optim/lora_optimizer.h
    operator/finetune_ops/optim/train_lora_gemma.cpp
  )
  local path
  local found=()
  for path in "${stale[@]}"; do
    if [ -e "$ROOT/$path" ]; then
      found+=("$path")
    fi
  done
  if [ "${#found[@]}" -gt 0 ]; then
    say "[FAIL] Stale operator API files remain in maintained tree:"
    printf '  %s\n' "${found[@]}"
    fail=1
  else
    say "[OK]   Stale operator API files are archived"
  fi
}

check_shell_syntax() {
  local scripts=(
    scripts/check_local_assets.sh
    scripts/check_lm_alignment_fixture.sh
    scripts/check_peft_lora_step_fixture.sh
    scripts/run_training_smoke.sh
    scripts/run_training_real_assets.sh
    scripts/lib/asset_paths.sh
    scripts/android/adb_resource_monitor.sh
    scripts/android/android_env.sh
    scripts/android/build_mft_sdk_aar.sh
    scripts/android/build_qwen_android.sh
    scripts/android/publish_mft_sdk_local.sh
    scripts/android/run_mft_sdk_device_smoke.sh
    scripts/android/run_qwen_qnli_native_phone.sh
    scripts/android/stage_qwen_qnli_phone_assets.sh
    examples/gpt2_small_lora_finetune/run_train.sh
    examples/gpt2_medium_lora_finetune/run_train.sh
    examples/gemma_3_270m_lora_finetune/run_train.sh
    examples/gemma_3_1b_pt_lora_finetune/run_train.sh
    examples/qwen_lora_finetune/run_wikitext.sh
    examples/qwen_lora_finetune/run_mmlu.sh
    examples/qwen_lora_finetune/run_qnli.sh
  )
  local path
  for path in "${scripts[@]}"; do
    bash -n "$ROOT/$path"
  done
  say "[OK]   Maintained shell entrypoints pass bash -n"
}

check_python_syntax() {
  local files=(
    scripts/asset_paths.py
    scripts/generate_lm_alignment_fixture.py
    scripts/generate_peft_lora_step_fixture.py
    scripts/generate_tokenizer_hf_golden_fixtures.py
    scripts/prepare_arcc_jsonl.py
    scripts/prepare_mcq_jsonl.py
    scripts/prepare_qnli_jsonl.py
    scripts/pretokenize_wikitext2_gemma.py
  )
  (
    cd "$ROOT"
    python3 - "${files[@]}" <<'PY'
import pathlib
import sys

for path in sys.argv[1:]:
    source = pathlib.Path(path).read_text(encoding="utf-8")
    compile(source, path, "exec")
PY
  )
  say "[OK]   Maintained Python entrypoints pass py_compile"
}

check_tokenizer_alignment_gate() {
  if [ "${MFT_RUN_TOKENIZER_ALIGNMENT:-}" != "1" ] &&
     [ -z "${MFT_MODEL_ROOT:-}" ] &&
     [ -z "${MOBILEFINETUNER_MODEL_ROOT:-}" ]; then
    say "[OK]   HF tokenizer alignment gate is available; set MFT_MODEL_ROOT to execute it"
    return
  fi

  say "[INFO] Running HF tokenizer alignment gate"
  if ! MFT_OPERATOR_BUILD_DIR="${MFT_OPERATOR_BUILD_DIR:-$ROOT/operator/build-release-audit}" \
       bash "$ROOT/scripts/check_tokenizer_hf_alignment.sh"; then
    say "[FAIL] HF tokenizer alignment gate failed"
    fail=1
  fi
}

check_no_personal_paths
check_no_archived_experiment_surface
check_large_source_files
check_package_files
check_no_cjk_text_in_maintained_tree
check_no_generated_caches
check_no_stale_operator_api
check_shell_syntax
check_python_syntax
check_tokenizer_alignment_gate

if [ "$fail" -ne 0 ]; then
  exit 1
fi

say "[OK]   Release tree audit passed"
