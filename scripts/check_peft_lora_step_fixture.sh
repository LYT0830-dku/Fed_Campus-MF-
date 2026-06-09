#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 runs/model_alignment/peft_lora_step_fixture.json" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
FIXTURE="$1"
if [ ! -f "$FIXTURE" ]; then
  echo "PEFT LoRA step fixture not found: $FIXTURE" >&2
  exit 1
fi

BUILD_DIR="${MFT_OPERATOR_BUILD_DIR:-$ROOT/operator/build}"
cmake -S "$ROOT/operator" -B "$BUILD_DIR" -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF
cmake --build "$BUILD_DIR" --target test_auto_model_peft_step_alignment -j"${JOBS:-4}"

MFT_PEFT_LORA_STEP_FIXTURE="$FIXTURE" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure -R AutoModelPEFTStepAlignment
