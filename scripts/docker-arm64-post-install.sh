#!/bin/bash
# arm64 post-install fixups for Docker builds.
set -e

echo "=== building flash-attn from source (sm_100 / GB200) ==="
# Run from /tmp so uv doesn't read pyproject.toml's [tool.uv.extra-build-variables]
# which sets FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE and prevents CUDA kernel compilation.
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
(cd /tmp && uv pip install --python /app/.venv/bin/python \
    "flash-attn==2.8.3" --no-build-isolation --no-binary flash-attn --no-cache)

echo "=== reinstalling flash-attn-cute (flash-attn overwrites it with a stub) ==="
uv pip install --reinstall --no-deps \
    "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@96bd151#subdirectory=flash_attn/cute"

echo "=== copying ampere_helpers.py from flash-attn-cute into cutlass-dsl ==="
SITE_PACKAGES=$(uv run --project /app --no-sync python -c 'import site; print(site.getsitepackages()[0])')
cp "$SITE_PACKAGES/flash_attn/cute/ampere_helpers.py" \
   "$SITE_PACKAGES/nvidia_cutlass_dsl/python_packages/cutlass/utils/ampere_helpers.py"
