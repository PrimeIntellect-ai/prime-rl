"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

# ---------------------------------------------------------------------------
# tilelang ships a stub libcudart that proxies to the real CUDA runtime via
# dlsym(RTLD_DEFAULT, ...). If the stub's symbols are the first ones found
# (because nothing has loaded the real libcudart globally yet) its self-check
# fails and the stub aborts — hit the moment any code calls into the
# classic-cudaMalloc MemPool (used for NIXL-registered slot buffers).
#
# Preloading the real library with RTLD_GLOBAL at this very early point —
# before transformers/torch/tilelang are pulled into the process — makes
# dlsym find the real symbols first.
#
# Wrapped in try/except because CDLL can fail on machines without a real
# CUDA runtime (e.g. CI containers).
# ---------------------------------------------------------------------------
import ctypes as _ctypes

try:
    _ctypes.CDLL("libcudart.so", mode=_ctypes.RTLD_GLOBAL)
except OSError:
    pass


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
