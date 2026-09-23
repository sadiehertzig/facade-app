#!/usr/bin/env bash
# Runs the app's small standalone tests on your Mac (no simulator, no models).
# Each test compiles a few app source files together with a test file that has
# its own main(), then runs it. A non-zero exit means a test failed.
#
#   app/ios/scripts/run_host_tests.sh
set -euo pipefail

IOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$IOS_DIR/FACADE/App"
TESTS="$IOS_DIR/Tests"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

run() {  # run <name> <swiftc arguments...>
  local name="$1"; shift
  echo "==> $name"
  xcrun swiftc -parse-as-library -O "$@" -o "$OUT/$name"
  "$OUT/$name"
}

run pixel-metrics \
  "$TESTS/PixelMetricsTests.swift" "$APP/PixelMetrics.swift"

run mini-photoguard -D MINI_PHOTOGUARD_STANDALONE_TEST \
  "$TESTS/MiniPhotoGuardEngineTests.swift" "$APP/MiniPhotoGuardEngine.swift"

run photo-preparation -I "$IOS_DIR/FACADE/Native/include" \
  "$TESTS/LivePhotoPreparationTest.swift" "$TESTS/LivePhotoPreparationStubs.swift" \
  "$APP/LivePhotoEngine.swift" "$APP/NativeImageIO.swift" "$APP/NativeSchedule.swift"

echo
echo "All host tests passed."
