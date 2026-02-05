#!/bin/bash
# Reinstalls flash-attn-cute to ensure it overwrites the older cute module from flash-attn
# Run this after `uv sync` if you have both flash-attn and flash-attn-cute extras enabled

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
