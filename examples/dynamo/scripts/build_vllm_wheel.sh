#!/usr/bin/env bash
set -euo pipefail

VLLM_REPO=${VLLM_REPO:-https://github.com/biswapanda/vllm.git}
VLLM_REV=${VLLM_REV:-e74fc3f}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/dist/dynamo}
BUILD_DIR=${BUILD_DIR:-$PWD/.build/vllm-dynamo}
MAX_JOBS=${MAX_JOBS:-$(nproc)}

mkdir -p "$OUTPUT_DIR" "$(dirname "$BUILD_DIR")"
if [[ ! -d "$BUILD_DIR/.git" ]]; then
  git clone "$VLLM_REPO" "$BUILD_DIR"
fi
git -C "$BUILD_DIR" fetch origin "$VLLM_REV"
git -C "$BUILD_DIR" checkout --detach FETCH_HEAD
actual_rev=$(git -C "$BUILD_DIR" rev-parse --short=7 HEAD)
[[ "$actual_rev" == "$VLLM_REV" ]] || {
  echo "Expected vLLM $VLLM_REV, got $actual_rev" >&2
  exit 1
}

(
  cd "$BUILD_DIR"
  uv venv --clear --python 3.12 .venv-build
  uv pip install --python .venv-build/bin/python -r requirements/build/cuda.txt
  MAX_JOBS="$MAX_JOBS" uv build --python .venv-build/bin/python     --wheel --no-build-isolation --out-dir "$OUTPUT_DIR"
)
