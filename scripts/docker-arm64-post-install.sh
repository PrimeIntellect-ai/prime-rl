#!/bin/bash
# arm64 post-install fixups for Docker builds.
set -e

echo "=== building flash-attn from source (sm_100 / GB200) ==="
# FLASH_ATTENTION_FORCE_BUILD=TRUE skips the prebuilt wheel download attempt
# and forces a local CUDA kernel compilation.
# FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE ensures the CUDA extension is compiled.
TORCH_CUDA_ARCH_LIST="10.0" MAX_JOBS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
    uv pip install "flash-attn==2.8.3" --no-build-isolation --no-binary flash-attn --no-cache

echo "=== reinstalling flash-attn-cute (flash-attn overwrites it with a stub) ==="
uv pip install --reinstall --no-deps \
    "flash-attn-cute @ git+https://github.com/Dao-AILab/flash-attention.git@e2743ab5#subdirectory=flash_attn/cute"

# TODO: remove once flash-attn gates the ampere_helpers import or cutlass-dsl re-adds it.
echo "=== copying ampere_helpers.py from flashinfer vendor ==="
SITE_PACKAGES=".venv/lib/python3.12/site-packages"
cp "$SITE_PACKAGES/flashinfer/data/cutlass/python/CuTeDSL/cutlass/utils/ampere_helpers.py" \
   "$SITE_PACKAGES/nvidia_cutlass_dsl/python_packages/cutlass/utils/ampere_helpers.py"
