#!/usr/bin/env bash
# Sync the PrimeRL environment and keep the aarch64 accelerator stack valid.
#
# On GH200/aarch64, FA2 currently resolves from an sdist unless a matching
# wheel/cache artifact already exists. Normal syncs must not silently spend an
# allocation rebuilding FA2. By default this wrapper makes such rebuilds fail
# fast; set PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1 only for an intentional compute-node
# repair/build.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_aarch64=false
if [ "$(uname -m)" = "aarch64" ]; then
    is_aarch64=true
fi

sync_args=(--extra all --extra envs --extra gpt-oss --extra modelexpress --group dev)
wheelhouse="${PRIME_RL_WHEELHOUSE:-$ROOT_DIR/wheels}"
if [ -d "$wheelhouse" ]; then
    sync_args+=(--find-links "$wheelhouse")
fi
if [ "$is_aarch64" = true ] && [ "${PRIME_RL_ALLOW_FLASH_ATTN_BUILD:-0}" != "1" ]; then
    sync_args+=(--no-build-package flash-attn)
fi
sync_args+=("$@")

if ! uv sync "${sync_args[@]}"; then
    if [ "$is_aarch64" = true ] && [ "${PRIME_RL_ALLOW_FLASH_ATTN_BUILD:-0}" != "1" ]; then
        cat >&2 <<'EOF'
ERROR: uv sync failed on aarch64 with flash-attn source builds disabled.

This is intentional. GH200 syncs must not silently rebuild flash-attn during
ordinary dependency reconciliation. Put a matching wheel in PRIME_RL_WHEELHOUSE
(default: ./wheels), or run an explicit compute-node repair/build:

  PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1 bash scripts/sync-prime-rl-env.sh

Do not run the explicit build on a login node.
EOF
    fi
    exit 1
fi

if [ "$is_aarch64" != true ]; then
    exit 0
fi

export UV_NO_SYNC=1
export VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"

if uv run --no-sync python - <<'PY'
from flash_attn import flash_attn_varlen_func as fa2
from flash_attn_interface import flash_attn_varlen_func as fa3
from flash_attn.cute import flash_attn_varlen_func as fa4
PY
then
    echo "FA2/FA3/FA4 import surface OK."
else
    echo "FA import surface incomplete; repairing FA4 namespace if needed..."
    bash scripts/fix-flash-attn-cute.sh
fi

if uv run --no-sync python - <<'PY'
from flash_attn import flash_attn_varlen_func as fa2
from flash_attn_interface import flash_attn_varlen_func as fa3
from flash_attn.cute import flash_attn_varlen_func as fa4
print("Verified FA2/FA3/FA4 import surface.")
PY
then
    exit 0
fi

if [ "${PRIME_RL_ALLOW_FLASH_ATTN_BUILD:-0}" != "1" ]; then
    cat >&2 <<'EOF'
ERROR: FA2/FA3/FA4 import surface is still invalid after FA4 namespace repair.

Refusing to rebuild flash-attn during a normal sync. Run an explicit compute-node
repair/build instead:

  PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1 bash scripts/sync-prime-rl-env.sh

Do not run the explicit build on a login node.
EOF
    exit 1
fi

echo "FA import surface incomplete; explicit build allowed, rebuilding aarch64 FlashAttention stack..."
bash scripts/docker-arm64-post-install.sh
