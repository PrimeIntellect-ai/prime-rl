"""Stub flash_attn.flash_attn_interface for ARM64 GB200 builds.

Exports the subset of names that ring_flash_attn (0.1.8) imports so that
eager `from flash_attn.flash_attn_interface import (...)` statements
succeed at module load. Any actual *call* into these raises — we rely on
vLLM + SDPA for the real attention backend.
"""


def _stub(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn.flash_attn_interface is stubbed on ARM64 GB200. "
        "Rebuild with docker-arm64-post-install.sh for real flash_attn."
    )


# Public API (some users reach for these)
flash_attn_func = _stub
flash_attn_varlen_func = _stub
flash_attn_qkvpacked_func = _stub
flash_attn_kvpacked_func = _stub
flash_attn_with_kvcache = _stub
flash_attn_unpadded_func = _stub
flash_attn_varlen_qkvpacked_func = _stub
flash_attn_varlen_kvpacked_func = _stub

# Private names imported by ring_flash_attn — must exist for module-load-time
# `from flash_attn.flash_attn_interface import (_flash_attn_varlen_forward, ...)`
_flash_attn_forward = _stub
_flash_attn_backward = _stub
_flash_attn_varlen_forward = _stub
_flash_attn_varlen_backward = _stub
