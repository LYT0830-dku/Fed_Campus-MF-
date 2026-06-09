#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
BUILD_DIR="${MFT_OPERATOR_BUILD_DIR:-$ROOT/operator/build}"
OUTPUT="${MFT_TOKENIZER_GOLDEN_JSONL:-$ROOT/runs/tokenizer_golden/hf_tokenizer_golden.jsonl}"
MODELS="${MFT_TOKENIZER_MODELS:-gpt2_small gpt2_medium qwen gemma_270m gemma_1b_pt}"

read -r -a MODEL_ARGS <<< "$MODELS"

python3 "$ROOT/scripts/generate_tokenizer_hf_golden_fixtures.py" \
  --strict \
  --models "${MODEL_ARGS[@]}" \
  --output "$OUTPUT"

if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
  cmake -S "$ROOT/operator" -B "$BUILD_DIR" -DBUILD_TESTS=ON
fi

cmake --build "$BUILD_DIR" --target test_tokenizer_hf_golden

MFT_TOKENIZER_GOLDEN_JSONL="$OUTPUT" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure -R TokenizerHFGolden
