"""Compatibility shim: ring_flash_attn + transformers >= 5.4.

ring_flash_attn 0.1.8 imports `is_flash_attn_greater_or_equal_2_10` from
`transformers.modeling_flash_attention_utils`. This symbol was removed from
that module in transformers 5.4 (still available as a deprecated function
in `transformers.utils.import_utils`, scheduled for removal in 5.8).

ring_flash_attn's except-branch is a no-op (imports the same symbol again),
so the import crashes on transformers >= 5.4. We patch the symbol back in as
`True` — the check is dead code since no one uses flash_attn < 2.1.0 anymore.

Upstream fix: https://github.com/zhuzilin/ring-flash-attention/pull/85
Remove this shim once ring_flash_attn ships a fixed version.
"""

import transformers.modeling_flash_attention_utils as _mfau

if not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
    # ring_flash_attn uses this as a bare value (if x:), but other code may
    # call it as a function (if x():). Use a callable that is also truthy.
    _mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True
