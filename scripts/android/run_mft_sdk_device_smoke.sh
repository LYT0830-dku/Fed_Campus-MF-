#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
ANDROID_DIR="$ROOT/android-visualizer"
source "$ROOT/scripts/android/android_env.sh"

if [ -n "${JAVA_HOME:-}" ] && [ ! -x "$JAVA_HOME/bin/java" ]; then
  unset JAVA_HOME
fi

if [ -z "${JAVA_HOME:-}" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || /usr/libexec/java_home)"
fi

if [ -z "${ANDROID_HOME:-}" ] && [ -d "$HOME/Library/Android/sdk" ]; then
  export ANDROID_HOME="$HOME/Library/Android/sdk"
fi

run_with_timeout() {
  local seconds="$1"
  shift

  "$@" &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$seconds" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "$pid"
}

ADB="$(mf_android_resolve_adb)"
APK="$ANDROID_DIR/sdk-sample/build/outputs/apk/debug/sdk-sample-debug.apk"
PKG="com.mobilefinetuner.sdk.sample"
ACTIVITY="$PKG/.MainActivity"
INSTALL_TIMEOUT_SECONDS="${MFT_SDK_INSTALL_TIMEOUT_SECONDS:-180}"

is_installed() {
  "$ADB" shell pm list packages | rg -q "^package:$PKG$"
}

install_apk() {
  local first_status=0
  run_with_timeout "$INSTALL_TIMEOUT_SECONDS" "$ADB" install -r -t "$APK" || first_status=$?
  if [ "$first_status" -eq 0 ]; then
    return 0
  fi

  if [ "$first_status" -eq 124 ]; then
    echo "[Android SDK] Streaming APK install timed out after ${INSTALL_TIMEOUT_SECONDS}s; retrying with --no-streaming." >&2
  else
    echo "[Android SDK] Streaming APK install failed with status $first_status; retrying with --no-streaming." >&2
  fi

  run_with_timeout "$INSTALL_TIMEOUT_SECONDS" "$ADB" install --no-streaming -r -t "$APK"
}

cd "$ANDROID_DIR"
./gradlew :sdk-sample:assembleDebug

"$ADB" devices
"$ADB" logcat -c

if [ "${MFT_SDK_SMOKE_SKIP_INSTALL:-0}" = "1" ]; then
  if ! is_installed; then
    echo "[Android SDK] MFT_SDK_SMOKE_SKIP_INSTALL=1 was set, but $PKG is not installed." >&2
    exit 1
  fi
else
  if [ "${MFT_SDK_SMOKE_FORCE_REINSTALL:-0}" = "1" ] && is_installed; then
    "$ADB" shell pm uninstall "$PKG" >/dev/null || true
  fi

  if ! install_apk; then
    if is_installed; then
      echo "[Android SDK] APK install did not complete, but $PKG is installed; continuing with launch smoke." >&2
    else
      echo "[Android SDK] APK install timed out. Unlock the phone and allow USB app installation, then retry." >&2
      exit 124
    fi
  fi
fi

"$ADB" shell am start -n "$ACTIVITY"
sleep 10

LOG_OUTPUT="$("$ADB" logcat -d -t 500)"
printf '%s\n' "$LOG_OUTPUT" | rg "MFTSdkSample|MobileFineTuner|Self-test" || true

if printf '%s\n' "$LOG_OUTPUT" | rg -q "MFTSdkSample.*Self-test passed"; then
  echo "[Android SDK] Device smoke passed"
elif "$ADB" shell uiautomator dump /sdcard/mft_sdk_smoke_ui.xml >/dev/null 2>&1 &&
     "$ADB" shell cat /sdcard/mft_sdk_smoke_ui.xml | rg -q "Self-test passed"; then
  "$ADB" shell cat /sdcard/mft_sdk_smoke_ui.xml | rg "MobileFineTuner|Self-test|loss=|trainable_tensors=|elapsed_ms=" || true
  echo "[Android SDK] Device smoke passed via UI dump"
else
  echo "[Android SDK] Device smoke did not find a self-test pass marker in logcat" >&2
  exit 1
fi
