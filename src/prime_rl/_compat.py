"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

import importlib.util

# ---------------------------------------------------------------------------
# ring_flash_attn + transformers >= 5.4
#
# ring_flash_attn 0.1.8 imports `is_flash_attn_greater_or_equal_2_10` from
# `transformers.modeling_flash_attention_utils`, removed in transformers 5.4.
# Some test environments intentionally do not install transformers, so skip
# cleanly when the package is absent but still fail loudly for broken installs.
#
# Upstream fix: https://github.com/zhuzilin/ring-flash-attention/pull/85
# Remove once ring_flash_attn ships a fixed version.
# ---------------------------------------------------------------------------
if importlib.util.find_spec("transformers") is not None:
    import transformers.modeling_flash_attention_utils as _mfau

    if not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
        _mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True

    # -----------------------------------------------------------------------
    # transformers >= 5.5 hub_kernels offline regression
    #
    # lazy_load_kernel() resolves kernel versions via HfApi().list_repo_refs(),
    # which raises OfflineModeIsEnabled when HF_HUB_OFFLINE=1 even if the
    # kernel is cached locally. Only FileNotFoundError and AssertionError are
    # caught upstream; OfflineModeIsEnabled is not.
    #
    # Remove once huggingface/transformers fixes lazy_load_kernel offline handling.
    # -----------------------------------------------------------------------
    try:
        import transformers.integrations.hub_kernels as _hub_kernels
    except ImportError:
        _hub_kernels = None  # transformers < 5.5, no patch needed

    if _hub_kernels is not None:
        from huggingface_hub import constants as _hf_constants
        from huggingface_hub.errors import OfflineModeIsEnabled

        _original_lazy_load_kernel = _hub_kernels.lazy_load_kernel

        def _patched_lazy_load_kernel(kernel_name, mapping=_hub_kernels._KERNEL_MODULE_MAPPING):
            try:
                return _original_lazy_load_kernel(kernel_name, mapping)
            except OfflineModeIsEnabled:
                # Return None so NemotronH skips hub kernels; prime-rl's
                # _patch_mamba2_use_triton_ssd() uses mamba_ssm directly.
                mapping[kernel_name] = None
                return None
            except ValueError as exc:
                if _hf_constants.is_offline_mode() and "could not verify publisher trust status" in str(exc):
                    mapping[kernel_name] = None
                    return None
                raise

        _hub_kernels.lazy_load_kernel = _patched_lazy_load_kernel
