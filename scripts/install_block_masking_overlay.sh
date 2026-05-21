#!/usr/bin/env bash
# Install block masking overlay onto the installed vLLM wheel.
#
# Strategy (same as Memento's install_overlay.sh):
#   1. Find the installed vLLM site-packages directory
#   2. Backup critical .so-interface files
#   3. Rsync overlay .py files over the installed vLLM
#   4. Restore .so-interface files (compiled extensions have mismatched signatures)
#   5. Verify imports work
#
# Usage:
#   scripts/install_block_masking_overlay.sh [--venv .venv]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERLAY_DIR="$REPO_DIR/overlays/vllm"
VENV="${1:-${UV_PROJECT_ENVIRONMENT:-.venv}}"

# Find vLLM site-packages
if [[ "$VENV" = /* ]]; then
    PYTHON="$VENV/bin/python"
else
    PYTHON="$REPO_DIR/$VENV/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

VLLM_DIR=$("$PYTHON" -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
if [[ ! -d "$VLLM_DIR" ]]; then
    echo "ERROR: vLLM not installed in $VENV"
    exit 1
fi
VLLM_VERSION=$("$PYTHON" -c "from importlib.metadata import version; print(version('vllm'))")

echo "=== Block Masking Overlay Installer ==="
echo "Overlay source: $OVERLAY_DIR"
echo "vLLM target:    $VLLM_DIR"
echo "vLLM version:   $VLLM_VERSION"
echo ""

if [[ -n "${EXPECTED_VLLM_VERSION_PREFIX:-}" ]]; then
    case "$VLLM_VERSION" in
        "${EXPECTED_VLLM_VERSION_PREFIX}"*) ;;
        *)
            echo "ERROR: Expected vLLM version prefix ${EXPECTED_VLLM_VERSION_PREFIX}, got ${VLLM_VERSION}"
            echo "Install the matching vLLM wheel or use a base image built from the current lockfile before applying overlays."
            exit 1
            ;;
    esac
fi

# Check overlay directory exists and has files
if [[ ! -d "$OVERLAY_DIR" ]]; then
    echo "ERROR: Overlay directory not found: $OVERLAY_DIR"
    exit 1
fi

NUM_FILES=$(find "$OVERLAY_DIR" -name "*.py" | wc -l | tr -d ' ')
if [[ "$NUM_FILES" -eq 0 ]]; then
    echo "ERROR: No .py files found in overlay directory"
    exit 1
fi
echo "Found $NUM_FILES overlay files"

if [[ -f "$OVERLAY_DIR/config/vllm.py" ]]; then
    echo "ERROR: Refusing to install stale full-file overlay: config/vllm.py"
    echo "Block masking config must be injected by the prime_rl vLLM plugin."
    exit 1
fi

# Backup .so-interface files that must be preserved.
# These compiled extension interfaces have argument counts that don't match
# if the overlay's Python wrappers are from a different upstream version.
BACKUP_DIR=$(mktemp -d)
echo "Backing up .so-interface files to $BACKUP_DIR"

INTERFACE_FILES=(
    "vllm_flash_attn/flash_attn_interface.py"
    "_custom_ops.py"
)

for f in "${INTERFACE_FILES[@]}"; do
    src="$VLLM_DIR/$f"
    if [[ -f "$src" ]]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$f")"
        cp "$src" "$BACKUP_DIR/$f"
        echo "  Backed up: $f"
    fi
done

# Ensure prime_rl.inference.block_masking is importable.
# If prime_rl is already installed (editable from /app/src/), just verify the
# block_masking directory exists in the source tree. If not, install into
# site-packages as a fallback.
SITE_PACKAGES=$(dirname "$VLLM_DIR")
BM_SRC="$REPO_DIR/src/prime_rl/inference/block_masking"

if "$PYTHON" -c "import prime_rl" 2>/dev/null; then
    echo "prime_rl already installed — ensuring block_masking is in source tree..."
    if [[ -d "$BM_SRC" ]] && [[ -f "$BM_SRC/__init__.py" ]]; then
        echo "  block_masking found at $BM_SRC"
    else
        echo "  WARNING: block_masking not found at $BM_SRC"
    fi
else
    echo "prime_rl not installed — installing block_masking into site-packages..."
    BM_DST="$SITE_PACKAGES/prime_rl/inference/block_masking"
    if [[ -d "$BM_SRC" ]]; then
        mkdir -p "$BM_DST"
        touch "$SITE_PACKAGES/prime_rl/__init__.py"
        touch "$SITE_PACKAGES/prime_rl/inference/__init__.py"
        cp "$BM_SRC"/*.py "$BM_DST/"
        echo "  Installed $(find "$BM_DST" -name '*.py' | wc -l | tr -d ' ') files to $BM_DST"
    else
        echo "  WARNING: Vendored block_masking not found at $BM_SRC"
    fi
fi

# Copy overlay .py files (preserving directory structure)
echo ""
echo "Installing overlay..."
if command -v rsync &>/dev/null; then
    rsync -av --include='*/' --include='*.py' --exclude='*' \
        "$OVERLAY_DIR/" "$VLLM_DIR/" 2>&1 | grep -v '/$' | head -30
else
    find "$OVERLAY_DIR" -name '*.py' -type f | while read -r src; do
        rel="${src#$OVERLAY_DIR/}"
        dst="$VLLM_DIR/$rel"
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "$rel"
    done
fi

# Restore .so-interface files
echo ""
echo "Restoring .so-interface files..."
for f in "${INTERFACE_FILES[@]}"; do
    backup="$BACKUP_DIR/$f"
    dst="$VLLM_DIR/$f"
    if [[ -f "$backup" ]]; then
        cp "$backup" "$dst"
        echo "  Restored: $f"
    fi
done

rm -rf "$BACKUP_DIR"

# Verify imports
echo ""
echo "Verifying installation..."
ERRORS=0

verify() {
    if "$PYTHON" -c "$1" 2>/dev/null; then
        echo "  OK: $2"
    else
        echo "  FAIL: $2"
        "$PYTHON" -c "$1" 2>&1 | tail -3 || true
        ERRORS=$((ERRORS + 1))
    fi
}

verify "from vllm.config.block_masking import BlockMaskingConfig" "BlockMaskingConfig import"
verify "from vllm.config import BlockMaskingConfig" "BlockMaskingConfig via config.__init__"
verify "from vllm.v1.core.block_masking import BlockMaskingState, BlockMaskingProcessor" "BlockMaskingState/Processor import"
verify "from vllm.v1.core.sched.output import SchedulerOutput; fields = SchedulerOutput.__dataclass_fields__; assert 'kv_copy_operations' in fields and 'block_masking_barrier' in fields" "SchedulerOutput has block masking fields"
verify "from vllm.v1.core.kv_cache_manager import KVCacheManager; assert hasattr(KVCacheManager, 'compact_kv_cache')" "KVCacheManager.compact_kv_cache"
verify "from vllm.distributed.kv_events import KVSlotCopy" "KVSlotCopy import"
verify "from vllm.v1.engine import SpanRemovalResult" "SpanRemovalResult import"
verify "from vllm.v1.worker.block_table import BlockTable; assert hasattr(BlockTable, 'truncate_row')" "BlockTable.truncate_row"
verify "from vllm.plugins import load_general_plugins; load_general_plugins(); from vllm.v1.worker.gpu_model_runner import GPUModelRunner; assert getattr(GPUModelRunner, '_prime_rl_block_masking_patched', False); assert hasattr(GPUModelRunner, '_execute_kv_copy_operations'); from vllm.v1.worker.gpu_input_batch import InputBatch; assert getattr(InputBatch, '_prime_rl_block_masking_patched', False); from vllm.v1.engine.core import EngineCore; assert getattr(EngineCore, '_prime_rl_block_masking_barrier_patched', False); from vllm.config import VllmConfig; assert hasattr(VllmConfig, 'block_masking_config')" "prime_rl production plugin installs block masking patches"

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "ERROR: $ERRORS import verification(s) failed"
    exit 1
fi

echo ""
echo "=== Overlay installed successfully ($NUM_FILES files) ==="
