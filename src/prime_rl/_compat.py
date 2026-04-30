"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

# ---------------------------------------------------------------------------
# TileLang libcudart_stub.so wins dlsym(RTLD_DEFAULT) on ARM64 GB200 builds
#
# TileLang ships a stub `libcudart_stub.so` that gets loaded into the process
# alongside the real CUDA runtime. Without an explicit RTLD_GLOBAL load of
# the real libcudart, downstream dlsym() lookups (e.g. `cudaDeviceReset`
# from FlashInfer's `CudaRTLibrary`) hit the stub first and fail with
# `AttributeError: undefined symbol: cudaDeviceReset`.
#
# PI's GLM-MoE-DSA path solves this by preloading at the top of
# `sparse_mla_{fwd,bwd}.py`, but plain Qwen3 doesn't import sparse_mla.
# vLLM's WorkerProc subprocesses also don't reliably inherit LD_PRELOAD
# from the entrypoint shell (uv/spawn() interaction).
#
# Fix: explicit ctypes load of the real libcudart with RTLD_GLOBAL right
# here, before any model/vLLM module imports. This wins the dlsym race.
# Wrapped in try/except so non-CUDA build environments (CI, dev) don't
# break.
# ---------------------------------------------------------------------------
import ctypes as _ctypes
import os as _os

_REAL_LIBCUDART_CANDIDATES = (
    "/usr/local/cuda/lib64/libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
)
for _candidate in _REAL_LIBCUDART_CANDIDATES:
    if _os.path.exists(_candidate):
        try:
            _ctypes.CDLL(_candidate, mode=_ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue


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
