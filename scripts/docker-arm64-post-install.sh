#!/bin/bash
# arm64 post-install fixups: rebuild flash-attn from source for the target GPU.
#
# Why this exists: pyproject.toml sets FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE to keep
# `uv sync` fast; on x86_64 it pins a prebuilt wheel to fill in the binary, but no
# such wheel exists for aarch64. Without this script, `import flash_attn` fails on
# aarch64 with `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`.
#
# Defaults preserve the existing Docker behavior (sm_100 / GB200). On a host with
# `nvidia-smi` available, the compute capability is auto-detected from the local
# GPU. Override via env vars if needed:
#   TORCH_CUDA_ARCH_LIST   e.g. 9.0 (Hopper), 10.0 (Blackwell)
#   VENV_PATH              path to the venv (default: $(pwd)/.venv)
#   MAX_JOBS               parallel nvcc jobs (default: 4)
set -euo pipefail

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    # Try to detect from the local GPU. Tolerate any failure mode (binary missing,
    # driver not loaded, Docker buildx without --gpus) and fall back to GB200.
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
    : "${TORCH_CUDA_ARCH_LIST:=10.0}"
fi
export TORCH_CUDA_ARCH_LIST

VENV_PATH="${VENV_PATH:-$(pwd)/.venv}"
if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo "ERROR: no python at $VENV_PATH/bin/python. Run from the project root or set VENV_PATH." >&2
    exit 1
fi

export MAX_JOBS="${MAX_JOBS:-4}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

echo "=== building flash-attn from source (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST, MAX_JOBS=$MAX_JOBS) ==="
echo "    target venv: $VENV_PATH"
# Run from /tmp so uv ignores the project's [tool.uv.extra-build-variables],
# which sets FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE and would prevent kernel compilation.
(cd /tmp && uv pip install --python "$VENV_PATH/bin/python" \
    "flash-attn==2.8.3" --no-build-isolation --no-binary flash-attn --no-cache --reinstall-package flash-attn)

echo "=== reinstalling flash-attn-cute (flash-attn overwrites it with a stub) ==="
uv pip install --python "$VENV_PATH/bin/python" --reinstall --no-deps \
    "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@96bd151#subdirectory=flash_attn/cute"
