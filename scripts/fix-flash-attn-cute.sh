#!/bin/bash
# Both `flash-attn` (FA2) and `flash-attn-cute` (FA4) ship a `flash_attn/cute/`
# sub-package.  The one from `flash-attn` is a tiny stub, while the one from
# `flash-attn-cute` contains the real FA4 kernels (>1000 lines in interface.py).
# When both extras are installed, `uv sync` may install `flash-attn` *after*
# `flash-attn-cute`, causing the stub to overwrite the real module.
#
# Additionally, FA4 imports `cutlass.utils.ampere_helpers` which nvidia-cutlass-dsl
# does not ship. We copy it from FA4's own source into the cutlass-dsl package.
#
# This script is idempotent: it checks first, and only reinstalls/patches if
# needed. Safe to call unconditionally after `uv sync`.

set -euo pipefail

FA4_SPEC="flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@96bd151#subdirectory=flash_attn/cute"

SITE_PACKAGES=$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')
CUTE_INTERFACE="$SITE_PACKAGES/flash_attn/cute/interface.py"

if [ -f "$CUTE_INTERFACE" ]; then
    LINES=$(wc -l < "$CUTE_INTERFACE")
else
    LINES=0
fi

if [ "$LINES" -gt 1000 ]; then
    echo "flash-attn-cute OK ($LINES lines at $CUTE_INTERFACE); no reinstall needed."
else
    echo "flash-attn-cute missing or clobbered ($LINES lines); reinstalling FA4..."
    uv pip install --python .venv/bin/python --reinstall --no-deps "$FA4_SPEC"
fi

AMPERE_SRC="$SITE_PACKAGES/flash_attn/cute/ampere_helpers.py"
AMPERE_DST="$SITE_PACKAGES/nvidia_cutlass_dsl/python_packages/cutlass/utils/ampere_helpers.py"

if [ -f "$AMPERE_DST" ]; then
    echo "ampere_helpers.py already present in cutlass-dsl; skipping."
elif [ -f "$AMPERE_SRC" ]; then
    cp "$AMPERE_SRC" "$AMPERE_DST"
    echo "Copied ampere_helpers.py from FA4 into cutlass-dsl."
else
    echo "ERROR: ampere_helpers.py not found in FA4 ($AMPERE_SRC)." >&2
    exit 1
fi

CUTE_INTERFACE=$(uv run --no-sync python -c 'import flash_attn.cute.interface as m; print(m.__file__)')
LINES=$(wc -l < "$CUTE_INTERFACE")
if [ "$LINES" -gt 1000 ]; then
    echo "Success: flash-attn-cute interface.py has $LINES lines (correct version)"
else
    echo "Error: flash-attn-cute interface.py has only $LINES lines (wrong version)"
    exit 1
fi
