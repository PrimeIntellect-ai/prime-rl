"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

# ---------------------------------------------------------------------------
# ring_flash_attn + transformers >= 5.4
#
# ring_flash_attn 0.1.8 imports `is_flash_attn_greater_or_equal_2_10` from
# `transformers.modeling_flash_attention_utils`, removed in transformers 5.4.
#
# Upstream fix: https://github.com/zhuzilin/ring-flash-attention/pull/85
# Remove once ring_flash_attn ships a fixed version.
# ---------------------------------------------------------------------------
import transformers.modeling_flash_attention_utils as _mfau

if not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
    _mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True
