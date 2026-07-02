"""Stub flash_attn_interface — raises on any real call.

Exports every symbol that ring_flash_attn, vLLM, and transformers import
from this module so the import chain doesn't break at module load time.
The actual NotImplementedError fires only if the function is *called*.
"""

_MSG = "flash_attn is stubbed on ARM64 GB200 — use attn='sdpa'"


def flash_attn_func(*a, **kw):
    raise NotImplementedError(_MSG)

def flash_attn_varlen_func(*a, **kw):
    raise NotImplementedError(_MSG)

def flash_attn_qkvpacked_func(*a, **kw):
    raise NotImplementedError(_MSG)

def flash_attn_kvpacked_func(*a, **kw):
    raise NotImplementedError(_MSG)

def flash_attn_with_kvcache(*a, **kw):
    raise NotImplementedError(_MSG)

def _flash_attn_forward(*a, **kw):
    raise NotImplementedError(_MSG)

def _flash_attn_backward(*a, **kw):
    raise NotImplementedError(_MSG)

def _flash_attn_varlen_forward(*a, **kw):
    raise NotImplementedError(_MSG)

def _flash_attn_varlen_backward(*a, **kw):
    raise NotImplementedError(_MSG)
