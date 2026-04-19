"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

import logging
import warnings

# ---------------------------------------------------------------------------
# Third-party warning noise on import
#
# - requests 2.33 asserts chardet < 6 but mini-swe-agent-plus → swebench pulls
#   in chardet 7.x, triggering RequestsDependencyWarning on every import.
# - tyro mis-handles Annotated[X | Y, Field(discriminator=...)] nested inside
#   another Union, producing a spurious TyroWarning for our AdvantageConfig.
#   See tyro/_resolver.py:unwrap_origin_strip_extras.
#
# Match by message pattern so we can install the filter before the offending
# package is imported (importing the warning class would trigger the warning).
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 \(.*\) or chardet \(.*\)/charset_normalizer \(.*\) doesn't match a supported version",
)
warnings.filterwarnings(
    "ignore",
    message=r".*does not match any type in Union.*",
)

# ---------------------------------------------------------------------------
# flash_attn.cute leaks DEBUG logs on import
#
# flash_attn/cute/cache_utils.py attaches its own StreamHandler and hardcodes
# logger.setLevel(DEBUG), so "Persistent cache disabled, using in-memory JIT
# cache" prints five times when flash_attn.cute.interface loads (once per
# get_jit_cache call in its module body). Those calls fire during the package
# import itself, so we can't downgrade the logger after-the-fact — install a
# drop-everything filter on the named logger before the package loads.
# ---------------------------------------------------------------------------
logging.getLogger("flash_attn.cute.cache_utils").addFilter(lambda r: False)

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


# ---------------------------------------------------------------------------
# transformers >= 5.5 hub_kernels offline regression
#
# lazy_load_kernel() resolves kernel versions via HfApi().list_repo_refs(),
# which raises OfflineModeIsEnabled when HF_HUB_OFFLINE=1 — even if the
# kernel is already cached locally. Only FileNotFoundError and AssertionError
# are caught; OfflineModeIsEnabled (a ConnectionError subclass) is not.
#
# This breaks Mamba-based models (NemotronH, Zamba2, Jamba, etc.) on SLURM
# worker nodes where HF_HUB_OFFLINE=1 is set.
#
# Fix: patch the except clause to also catch ConnectionError.
# Remove once huggingface/transformers fixes lazy_load_kernel offline handling.
# ---------------------------------------------------------------------------
try:
    import transformers.integrations.hub_kernels as _hub_kernels
except ImportError:
    _hub_kernels = None  # transformers < 5.5, no patch needed

if _hub_kernels is not None:
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

    _hub_kernels.lazy_load_kernel = _patched_lazy_load_kernel
