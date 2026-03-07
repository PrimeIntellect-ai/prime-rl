#!/bin/bash
# arm64 post-install fixups for Docker builds.
set -e

echo "=== installing flash-attn for arm64 ==="
# flash-attn setup.py has a CachedWheelsCommand that downloads a prebuilt aarch64
# wheel from GitHub releases (includes compiled flash_attn_2_cuda.so).
# Do NOT use --no-binary or --no-cache as those force a source build which is
# slow and hits FLASH_ATTENTION_SKIP_CUDA_BUILD from pyproject.toml.
uv pip install "flash-attn==2.8.3" --no-build-isolation

echo "=== reinstalling flash-attn-cute (flash-attn overwrites it with a stub) ==="
uv pip install --reinstall --no-deps \
    "flash-attn-cute @ git+https://github.com/Dao-AILab/flash-attention.git@e2743ab5#subdirectory=flash_attn/cute"

# TODO: remove once flash-attn gates the ampere_helpers import or cutlass-dsl re-adds it.
echo "=== copying ampere_helpers.py from flashinfer vendor ==="
SITE_PACKAGES=".venv/lib/python3.12/site-packages"
cp "$SITE_PACKAGES/flashinfer/data/cutlass/python/CuTeDSL/cutlass/utils/ampere_helpers.py" \
   "$SITE_PACKAGES/nvidia_cutlass_dsl/python_packages/cutlass/utils/ampere_helpers.py"
