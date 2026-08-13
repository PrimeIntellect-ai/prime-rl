#!/usr/bin/env bash
set -euo pipefail

DYNAMO_REPO=${DYNAMO_REPO:-https://github.com/ai-dynamo/dynamo.git}
DYNAMO_REV=${DYNAMO_REV:-fc556d9}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/dist/dynamo}
BUILD_DIR=${BUILD_DIR:-$PWD/.build/dynamo-prime-rl}

mkdir -p "$OUTPUT_DIR" "$(dirname "$BUILD_DIR")"
if [[ ! -d "$BUILD_DIR/.git" ]]; then
  git clone "$DYNAMO_REPO" "$BUILD_DIR"
fi
git -C "$BUILD_DIR" fetch origin "$DYNAMO_REV"
git -C "$BUILD_DIR" checkout --detach FETCH_HEAD
actual_rev=$(git -C "$BUILD_DIR" rev-parse --short=7 HEAD)
[[ "$actual_rev" == "$DYNAMO_REV" ]] || {
  echo "Expected Dynamo $DYNAMO_REV, got $actual_rev" >&2
  exit 1
}

(
  cd "$BUILD_DIR"
  uv build --wheel --out-dir "$OUTPUT_DIR"
  uvx --from 'maturin[patchelf]' maturin build     --release     --manifest-path lib/bindings/python/Cargo.toml     --out "$OUTPUT_DIR"
  cargo build --release --locked -p dynamo-vllm-sidecar
  install -m 0755 target/release/dynamo-vllm-sidecar "$OUTPUT_DIR/dynamo-vllm-sidecar"
)
