"""DeepSeek V4 compressor pooling through the cuDNN frontend's fused CSA/HCA compressor kernels.

`dsv4_compressor_pool` computes `C[e,d] = sum_s softmax_s(gate[e,s,d] + position_bias[s,d]) * kv[e,s,d]` over each
entry's pooling window as `DeepseekV4Compressor` lays it out, reading the window straight from the packed per-token
projections. One kernel runs each direction. The softmax weights and products stay in fp32 and the entry is rounded
to bf16 once, as vLLM's compressor keeps them in fp32 up to its RMSNorm.

cuDNN validates the kernels on SM100 and newer only. They use no TMA or tcgen05 instructions and match an fp64
reference on SM90 as well, so `_compressor_api` lowers cuDNN's capability floor to SM90.
"""

import functools
import importlib.util

import torch
from torch import Tensor

MIN_CAPABILITY_MAJOR = 9


def cudnn_pooling_supported(compress_rate: int, head_dim: int) -> bool:
    """Whether the cuDNN kernels cover this compressor shape on the current GPU."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < MIN_CAPABILITY_MAJOR:
        return False
    if importlib.util.find_spec("cudnn") is None or importlib.util.find_spec("cudnn.csa") is None:
        return False
    return compress_rate in (2, 4) or (compress_rate == 128 and head_dim in (128, 512))


@functools.cache
def _compressor_api():
    import cudnn.csa.compressor.api as api

    api.MINIMUM_COMPUTE_CAPABILITY_MAJOR = min(api.MINIMUM_COMPUTE_CAPABILITY_MAJOR, MIN_CAPABILITY_MAJOR)
    return api


@torch.library.custom_op("prime_rl::dsv4_compressor_pool", mutates_args=())
def _dsv4_compressor_pool(
    kv: Tensor,
    gate: Tensor,
    position_bias: Tensor,
    cu_seqlens: Tensor,
    cu_entries: Tensor,
    compress_rate: int,
    n_series: int,
    n_entries: int,
) -> Tensor:
    return _compressor_api().csa_compressor_forward_wrapper(
        kv.contiguous(),
        gate.contiguous(),
        position_bias.contiguous(),
        cu_seqlens,
        cu_entries,
        ratio=compress_rate,
        head_dim=kv.shape[1] // n_series,
        coff=n_series,
        total_comp=n_entries,
    )["out"]


@_dsv4_compressor_pool.register_fake
def _dsv4_compressor_pool_fake(
    kv: Tensor,
    gate: Tensor,
    position_bias: Tensor,
    cu_seqlens: Tensor,
    cu_entries: Tensor,
    compress_rate: int,
    n_series: int,
    n_entries: int,
) -> Tensor:
    return kv.new_empty(n_entries, kv.shape[1] // n_series)


@torch.library.custom_op("prime_rl::dsv4_compressor_pool_backward", mutates_args=())
def _dsv4_compressor_pool_backward(
    grad: Tensor,
    kv: Tensor,
    gate: Tensor,
    position_bias: Tensor,
    cu_seqlens: Tensor,
    cu_entries: Tensor,
    compress_rate: int,
    n_series: int,
) -> tuple[Tensor, Tensor, Tensor]:
    grads = _compressor_api().csa_compressor_backward_wrapper(
        kv.contiguous(),
        gate.contiguous(),
        position_bias.contiguous(),
        cu_seqlens,
        cu_entries,
        grad.contiguous(),
        ratio=compress_rate,
        head_dim=kv.shape[1] // n_series,
        coff=n_series,
    )
    return grads["grad_kv"], grads["grad_score"], grads["grad_ape"]


@_dsv4_compressor_pool_backward.register_fake
def _dsv4_compressor_pool_backward_fake(
    grad: Tensor,
    kv: Tensor,
    gate: Tensor,
    position_bias: Tensor,
    cu_seqlens: Tensor,
    cu_entries: Tensor,
    compress_rate: int,
    n_series: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(kv, memory_format=torch.contiguous_format),
        torch.empty_like(gate, memory_format=torch.contiguous_format),
        torch.empty_like(position_bias, memory_format=torch.contiguous_format),
    )


def _dsv4_compressor_pool_setup_context(ctx, inputs, output) -> None:
    kv, gate, position_bias, cu_seqlens, cu_entries, compress_rate, n_series, _ = inputs
    ctx.save_for_backward(kv, gate, position_bias, cu_seqlens, cu_entries)
    ctx.compress_rate = compress_rate
    ctx.n_series = n_series


def _dsv4_compressor_pool_autograd_backward(ctx, grad: Tensor):
    kv, gate, position_bias, cu_seqlens, cu_entries = ctx.saved_tensors
    grad_kv, grad_gate, grad_position_bias = _dsv4_compressor_pool_backward(
        grad, kv, gate, position_bias, cu_seqlens, cu_entries, ctx.compress_rate, ctx.n_series
    )
    return grad_kv, grad_gate, grad_position_bias, None, None, None, None, None


_dsv4_compressor_pool.register_autograd(
    _dsv4_compressor_pool_autograd_backward, setup_context=_dsv4_compressor_pool_setup_context
)


def dsv4_compressor_pool(
    kv: Tensor,
    gate: Tensor,
    position_bias: Tensor,
    cu_seqlens: Tensor,
    cu_entries: Tensor,
    compress_rate: int,
    n_series: int,
    n_entries: int,
) -> Tensor:
    """Pool one packed row's `(tokens, n_series * head_dim)` bf16 projections into `(n_entries, head_dim)` bf16 entries.

    `position_bias` is `(compress_rate, n_series * head_dim)` in any float dtype, and its gradient comes back in that
    dtype. `cu_seqlens` and `cu_entries` are int32 `(n_docs + 1,)` offsets of each document's first token and first
    entry, ending in the totals.
    """
    return _dsv4_compressor_pool(
        kv, gate, position_bias.float(), cu_seqlens, cu_entries, compress_rate, n_series, n_entries
    )
