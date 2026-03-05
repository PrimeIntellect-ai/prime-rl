#!/bin/bash
# Post-install fixups for arm64 Docker builds.
# On x86_64 this script is a no-op.
#
# Why arm64 needs special handling:
#   - flash-attn doesn't publish arm64 wheels, so we build from source
#   - Installing flash-attn overwrites flash_attn/cute/ with a stub,
#     so flash-attn-cute must be reinstalled afterwards
#   - nvidia-cutlass-dsl 4.4.1 dropped ampere_helpers.py but flash-attn
#     still imports it unconditionally (not gated by GPU arch)
set -e

if [ "$(uname -m)" != "aarch64" ]; then
    echo "Not arm64, skipping post-install fixups."
    exit 0
fi

echo "=== arm64 post-install: building flash-attn from source ==="
# Only target sm_100 (Blackwell) since arm64 clusters are GB200.
TORCH_CUDA_ARCH_LIST="10.0" MAX_JOBS=4 \
    uv pip install flash-attn --no-build-isolation

echo "=== arm64 post-install: reinstalling flash-attn-cute ==="
# flash-attn ships a stub flash_attn/cute/ that overwrites the real FA4
# kernels from flash-attn-cute. Reinstall to restore them.
uv pip install --reinstall --no-deps \
    "flash-attn-cute @ git+https://github.com/Dao-AILab/flash-attention.git@2b5db43#subdirectory=flash_attn/cute"

echo "=== arm64 post-install: ampere_helpers.py workaround ==="
# nvidia-cutlass-dsl 4.4.1 removed ampere_helpers.py, but flash-attn 2.8.3
# unconditionally imports it (flash_attn.cute.flash_fwd -> cutlass.utils.ampere_helpers).
# flashinfer vendors a copy, so we use that.
# TODO: remove once flash-attn gates the import or cutlass-dsl re-adds the file.
SITE_PACKAGES=".venv/lib/python3.12/site-packages"
cp "$SITE_PACKAGES/flashinfer/data/cutlass/python/CuTeDSL/cutlass/utils/ampere_helpers.py" \
   "$SITE_PACKAGES/nvidia_cutlass_dsl/python_packages/cutlass/utils/ampere_helpers.py"

echo "=== arm64 post-install: done ==="
