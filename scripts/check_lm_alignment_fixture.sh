#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
BUILD_DIR="${MFT_OPERATOR_BUILD_DIR:-$ROOT/operator/build}"
FIXTURE="${1:-${MFT_LM_ALIGNMENT_FIXTURE:-}}"

if [ -z "$FIXTURE" ]; then
  printf '[ERROR] Provide an LM alignment fixture path as argv[1] or MFT_LM_ALIGNMENT_FIXTURE.\n' >&2
  printf 'Generate one with scripts/generate_lm_alignment_fixture.py.\n' >&2
  exit 2
fi

if [ ! -f "$FIXTURE" ]; then
  printf '[ERROR] LM alignment fixture does not exist: %s\n' "$FIXTURE" >&2
  exit 2
fi

if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
  cmake -S "$ROOT/operator" -B "$BUILD_DIR" -DBUILD_TESTS=ON
fi

cmake --build "$BUILD_DIR" --target test_auto_model_alignment

MFT_LM_ALIGNMENT_FIXTURE="$FIXTURE" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure -R AutoModelAlignment
