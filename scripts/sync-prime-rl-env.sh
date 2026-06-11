#!/usr/bin/env bash
# Sync the PrimeRL environment and keep the aarch64 accelerator stack valid.
#
# On GH200/aarch64, FA2 is locked to a registry sdist, so any sync that wants
# to touch it implies an nvcc build. Normal syncs must not silently spend an
# allocation (or hammer a login node) rebuilding FA2: by default this wrapper
# excludes flash-attn from the locked sync and reconciles it against a
# pre-built wheel in PRIME_RL_WHEELHOUSE (default: ./wheels), failing loud if
# the wheelhouse lacks the locked version. Set PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1
# only for an intentional compute-node repair/build.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_aarch64=false
if [ "$(uname -m)" = "aarch64" ]; then
    is_aarch64=true
fi

sync_args=(--extra all --extra envs --extra gpt-oss --extra modelexpress --group dev)
wheelhouse="${PRIME_RL_WHEELHOUSE:-$ROOT_DIR/wheels}"
skip_fa_build=false
if [ "$is_aarch64" = true ] && [ "${PRIME_RL_ALLOW_FLASH_ATTN_BUILD:-0}" != "1" ]; then
    skip_fa_build=true
    # A locked sync can never satisfy the registry sdist pin from a local
    # wheel, so exclude flash-attn from the sync entirely and reconcile it
    # against the wheelhouse below. NOTE: when uv plans a reinstall,
    # --no-install-package suppresses only the install half — the uninstall
    # half still runs — which is why the wheelhouse restore is unconditional
    # on installed-vs-locked state, not on whether the sync "touched" it.
    sync_args+=(--no-install-package flash-attn)
fi
sync_args+=("$@")

uv sync "${sync_args[@]}"

if [ "$is_aarch64" != true ]; then
    exit 0
fi

export UV_NO_SYNC=1
export VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"

if [ "$skip_fa_build" = true ]; then
    # uv export evaluates the lock's platform markers, so this picks the
    # flash-attn entry that actually applies here (the lock carries one
    # entry per platform).
    if ! locked_fa_version="$(uv export --frozen --extra all --extra envs --extra gpt-oss --extra modelexpress --group dev --no-hashes --no-annotate --no-header | sed -n 's/^flash-attn==\([^ ;]*\).*/\1/p' | head -1)" \
        || [ -z "$locked_fa_version" ]; then
        echo "ERROR: could not read the locked flash-attn version from uv.lock (uv export)." >&2
        exit 1
    fi
    # `uv pip show` exits nonzero when the package is absent — the very case
    # the restore below handles — so don't let it abort the script.
    installed_fa_version="$(uv pip show --python "$VENV_PATH" flash-attn 2>/dev/null | awk '/^Version:/ { print $2 }' || true)"
    if [ "$installed_fa_version" != "$locked_fa_version" ]; then
        wheel="$(ls "$wheelhouse"/flash_attn-"$locked_fa_version"-*.whl 2>/dev/null | head -1)"
        if [ -z "$wheel" ]; then
            cat >&2 <<EOF
ERROR: flash-attn is at '${installed_fa_version:-<not installed>}' but uv.lock wants ${locked_fa_version},
and no matching wheel exists in the wheelhouse (${wheelhouse}).

This is intentional. GH200 syncs must not silently rebuild flash-attn during
ordinary dependency reconciliation. Put a flash_attn-${locked_fa_version}-*.whl in
PRIME_RL_WHEELHOUSE (default: ./wheels), or run an explicit compute-node build:

  PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1 bash scripts/sync-prime-rl-env.sh

Do not run the explicit build on a login node.
EOF
            exit 1
        fi
        echo "Restoring flash-attn ${locked_fa_version} from wheelhouse: $wheel"
        uv pip install --python "$VENV_PATH" "$wheel"
    fi
fi

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
