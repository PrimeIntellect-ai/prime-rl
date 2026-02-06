#!/bin/bash
# Both `flash-attn` (FA2) and `flash-attn-cute` (FA4) ship a `flash_attn/cute/`
# sub-package.  The one from `flash-attn` is a tiny stub, while the one from
# `flash-attn-cute` contains the real FA4 kernels (>1000 lines in interface.py).
# When both extras are installed, `uv sync` may install `flash-attn` *after*
# `flash-attn-cute`, causing the stub to overwrite the real module.
#
# This script reinstalls `flash-attn-cute` so the real module wins.
# Run it after `uv sync` if you have both flash-attn and flash-attn-cute extras enabled.

set -e

echo "Reinstalling flash-attn-cute to fix namespace conflict with flash-attn..."
uv pip install --reinstall --no-deps "flash-attn-cute @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=flash_attn/cute"

# Verify installation
LINES=$(wc -l < "$(python -c 'import flash_attn.cute.interface as m; print(m.__file__)')")
if [ "$LINES" -gt 1000 ]; then
    echo "Success: flash-attn-cute interface.py has $LINES lines (correct version)"
else
    echo "Error: flash-attn-cute interface.py has only $LINES lines (wrong version)"
    exit 1
fi
