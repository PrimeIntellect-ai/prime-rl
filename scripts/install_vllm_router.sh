#!/bin/bash
# Builds the patched vllm-router wheel (Rust via setuptools-rust) from our fork.
# Requires: Rust toolchain (cargo), Python 3.8+, and uv or pip.
# Override output dir: VLLM_ROUTER_WHEEL_DIR=/path/to/dir
set -euo pipefail

REPO_URL="https://github.com/S1ro1/router.git"
BRANCH="fix/preserve-extra-fields-disagg"
BUILD_DIR="/tmp/vllm-router-build"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${VLLM_ROUTER_WHEEL_DIR:-$SCRIPT_DIR/deps}"

echo "Cloning $REPO_URL (branch: $BRANCH)..."
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$BUILD_DIR"

echo "Building wheel -> $OUTPUT_DIR ..."
mkdir -p "$OUTPUT_DIR"
if command -v uv >/dev/null 2>&1; then
  uv build --wheel --out-dir "$OUTPUT_DIR" "$BUILD_DIR"
else
  (cd "$BUILD_DIR" && python3 -m pip wheel . --no-deps -w "$OUTPUT_DIR")
fi

echo "Cleaning up..."
rm -rf "$BUILD_DIR"

wheel=$(ls -1t "$OUTPUT_DIR"/*.whl 2>/dev/null | head -1)
echo "Done. Wheel: $wheel"
echo "Install: uv pip install $wheel"
