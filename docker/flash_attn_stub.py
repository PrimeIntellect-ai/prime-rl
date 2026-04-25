"""Stub flash_attn module for ARM64 GB200 builds that skip flash-attn
from-source compile.

Satisfies `import flash_attn` / `from flash_attn.flash_attn_interface import ...`
so that `ring_flash_attn` and other code paths that eagerly import
flash_attn don't fail at module load time. Any actual *use* of these
functions will raise NotImplementedError.

For scenario A (Qwen3-0.6B reverse-text, no context parallelism, no MoE),
flash_attn is not actually called — we just need the import to succeed.
Production use of MoE/CP requires a real flash-attn build (see
scripts/docker-arm64-post-install.sh).
"""

__version__ = "0.0.0-stub-for-arm64-gb200"


def _stub(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn is stubbed in this ARM64 GB200 image to avoid a 3-5 hr "
        "QEMU build. Rebuild with docker-arm64-post-install.sh enabled to "
        "get real flash_attn."
    )


# Fill out a few commonly-called names so `from flash_attn import X` works.
flash_attn_func = _stub
flash_attn_varlen_func = _stub
flash_attn_qkvpacked_func = _stub
flash_attn_kvpacked_func = _stub
flash_attn_with_kvcache = _stub
