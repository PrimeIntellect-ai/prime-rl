from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import triton
import triton.language as tl
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.activations import Silu

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


MegaMoePrecision = Literal["bf16", "mxfp8"]
_MEGA_MOE_KERNELS: dict[str, tuple[str, ...]] = {
    "bf16": ("bf16_mega_moe", "bf16_mega_moe_backward"),
    "mxfp8": ("fp8_fp4_mega_moe", "fp8_mega_moe_backward"),
}
_MEGA_MOE_MMA_TYPE: dict[str, str] = {"bf16": "bf16xbf16", "mxfp8": "fp8xfp8"}
_MX_BLOCK = 32


def mega_moe_available(precision: MegaMoePrecision = "bf16") -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() < (10, 0):
        return False
    # Upstream DeepGEMM has no training backward - only the prime-mega-moe build exposes both kernels
    return all(hasattr(deep_gemm, name) for name in _MEGA_MOE_KERNELS[precision])


def check_mega_moe_dims(hidden: int, intermediate_hidden: int) -> None:
    if hidden % 256 or intermediate_hidden % 128:
        raise ValueError(
            f"Mega MoE requires hidden ({hidden}) to be a multiple of 256 and intermediate_hidden "
            f"({intermediate_hidden}) a multiple of 128."
        )


@dataclass
class MegaMoeExpertWeights:
    l1: torch.Tensor
    l2: torch.Tensor


def reserve_sms_for_comm(num_reserved_sms: int) -> None:
    import deep_gemm

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    deep_gemm.set_num_sms(max(total - num_reserved_sms, 1))


_BUFFER_CACHE: dict[tuple, object] = {}
_BUFFER_REGISTRY: dict[int, object] = {}


def register_mega_moe_buffer(buffer) -> int:
    key = id(buffer)
    _BUFFER_REGISTRY[key] = buffer
    return key


def build_mega_moe_buffer(
    group: ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate_hidden: int,
    precision: MegaMoePrecision = "bf16",
):
    import deep_gemm

    check_mega_moe_dims(hidden, intermediate_hidden)
    mma_type = _MEGA_MOE_MMA_TYPE[precision]
    key = (id(group), num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden, mma_type)
    buffer = _BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden, mma_type=mma_type
        )
        _BUFFER_CACHE[key] = buffer
    return buffer


def _stage_routing(buffer, num_tokens: int, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    buffer.topk_idx[:num_tokens].copy_(topk_idx.view(num_tokens, buffer.num_topk))
    buffer.topk_weights[:num_tokens].copy_(topk_weights.view(num_tokens, buffer.num_topk))


def _stage_inputs(buffer, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    buffer.x[:num_tokens].copy_(x)
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)


# --- MXFP8 quantization -------------------------------------------------------------------------
# MX scale blocks run along the contraction axis, so every operand of the backward needs a copy
# quantized along the axis that GEMM contracts over: the forward operands (scaled along K of the
# forward GEMMs) serve the Z recompute, while `dX` and `dz = dy @ W2` need the weights scaled
# along 2I and H respectively. The kernels take packed UE8M0 scales in DeepGEMM's MN-major TMA
# layout with the UTCCP 128-row transpose applied; L1 rows are gate/up interleaved at 8 rows.


def _mx_cast(x: torch.Tensor, along_rows: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 cast of a 2D tensor with a 1x32 power-of-two scale per block, keeping ``x``'s storage
    layout. ``along_rows=False`` scales 32-wide blocks along the last dim (scales ``[rows, cols/32]``),
    ``along_rows=True`` along the first dim (scales ``[cols, rows/32]``)."""
    from prime_rl.trainer.models.kernels.fp8_utils import _per_token_fp8_kernel, ceil_div

    assert x.dim() == 2
    rows, cols = x.shape
    out = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=x.device)
    if along_rows:
        assert rows % _MX_BLOCK == 0
        sf = torch.empty((cols, rows // _MX_BLOCK), dtype=torch.float32, device=x.device)
        logical = (cols, rows)
        x_strides, out_strides = (x.stride(1), x.stride(0)), (out.stride(1), out.stride(0))
    else:
        assert cols % _MX_BLOCK == 0
        sf = torch.empty((rows, cols // _MX_BLOCK), dtype=torch.float32, device=x.device)
        logical = (rows, cols)
        x_strides, out_strides = (x.stride(0), x.stride(1)), (out.stride(0), out.stride(1))
    grid = lambda meta: (ceil_div(logical[0], meta["BLOCK_M"]), ceil_div(logical[1], meta["BLOCK_K"]))  # noqa: E731
    _per_token_fp8_kernel[grid](
        x,
        out,
        sf,
        logical[0],
        logical[1],
        *x_strides,
        *out_strides,
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=True,
        BLOCK_M=32,
        BLOCK_K=_MX_BLOCK,
        num_warps=4,
    )
    return out, sf


def _pack_ue8m0_k_major(sf: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scales ``[m, k/32]`` -> packed UE8M0 ``int32`` ``[m, k/128]`` (K-major)."""
    assert sf.dtype == torch.float32 and sf.size(-1) % 4 == 0
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def _mx_weight_sf(sf: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    """fp32 scales ``[E, mn, k/32]`` -> packed, MN-major, TMA-aligned, UTCCP-transposed ``int32``."""
    import deep_gemm
    from deep_gemm.mega import _transpose_sf_for_utccp

    num_groups = sf.shape[0]
    return _transpose_sf_for_utccp(deep_gemm.transform_sf_into_required_layout(sf, mn, k, (1, _MX_BLOCK), num_groups))


@triton.jit
def _mx_cast_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
    cols,
    stride_xe,
    stride_xr,
    stride_xc,
    stride_oe,
    stride_or,
    stride_oc,
    stride_se,
    stride_sc,
    stride_sk,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Per-expert cast of ``x[e]`` (``[R, C]``) with one power-of-two scale per 32 rows of each
    column; reads and writes are coalesced along C. ``sf[e, c, r / 32]``."""
    e = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_r = tl.program_id(axis=2)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r64, c64 = r.to(tl.int64), c.to(tl.int64)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    x = tl.load(x_ptr + e * stride_xe + r64[:, None] * stride_xr + c64[None, :] * stride_xc, mask=mask, other=0.0).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10)
    scale = tl.exp2(tl.ceil(tl.log2(tl.math.div_rn(amax, FP8_MAX))))
    y = tl.clamp(tl.math.div_rn(x, scale[None, :]), -FP8_MAX, FP8_MAX)
    tl.store(
        out_ptr + e * stride_oe + r64[:, None] * stride_or + c64[None, :] * stride_oc, y.to(tl.float8e4nv), mask=mask
    )
    tl.store(sf_ptr + e * stride_se + c64 * stride_sc + pid_r * stride_sk, scale, mask=c < cols)


def _cast_weight_along_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert ``[E, R, C]`` cast scaled along R; returns FP8 in the same layout and fp32 scales ``[E, C, R/32]``."""
    from prime_rl.trainer.models.kernels.fp8_utils import FP8_MAX, ceil_div

    num_experts, rows, cols = w.shape
    assert rows % _MX_BLOCK == 0
    out = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    sf = torch.empty((num_experts, cols, rows // _MX_BLOCK), dtype=torch.float32, device=w.device)
    block_c = 128
    grid = (num_experts, ceil_div(cols, block_c), rows // _MX_BLOCK)
    _mx_cast_rows_kernel[grid](
        w,
        out,
        sf,
        rows,
        cols,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sf.stride(0),
        sf.stride(1),
        sf.stride(2),
        FP8_MAX=FP8_MAX,
        BLOCK_R=_MX_BLOCK,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out, sf


@dataclass
class MegaMoeFP8ExpertWeights:
    l1: tuple[torch.Tensor, torch.Tensor]
    """Interleaved ``[E, 2I, H]`` scaled along H (forward L1 operand, backward Z recompute)."""
    l2: tuple[torch.Tensor, torch.Tensor] | None
    """``[E, H, I]`` scaled along I (forward L2 operand)."""
    l1_t: tuple[torch.Tensor, torch.Tensor] | None
    """Interleaved ``[E, 2I, H]`` scaled along 2I (backward dX)."""
    l2_t: tuple[torch.Tensor, torch.Tensor] | None
    """``[E, H, I]`` scaled along H (backward dz = dy @ W2)."""


def _fp8_weights(gate_up_proj: torch.Tensor, down_proj: torch.Tensor, for_backward: bool) -> MegaMoeFP8ExpertWeights:
    from deep_gemm.mega import _interleave_weights

    num_experts, two_i, hidden = gate_up_proj.shape
    inter = two_i // 2
    l1_bf16 = _interleave_weights(gate_up_proj.contiguous())
    l1_fp8, l1_sf = _mx_cast(l1_bf16.view(num_experts * two_i, hidden), along_rows=False)
    l1 = (l1_fp8.view(num_experts, two_i, hidden), _mx_weight_sf(l1_sf.view(num_experts, two_i, -1), two_i, hidden))
    if not for_backward:
        down = down_proj.contiguous()
        l2_fp8, l2_sf = _mx_cast(down.view(num_experts * hidden, inter), along_rows=False)
        l2 = (
            l2_fp8.view(num_experts, hidden, inter),
            _mx_weight_sf(l2_sf.view(num_experts, hidden, -1), hidden, inter),
        )
        return MegaMoeFP8ExpertWeights(l1=l1, l2=l2, l1_t=None, l2_t=None)
    l1_t_fp8, l1_t_sf = _cast_weight_along_rows(l1_bf16)
    l1_t = (l1_t_fp8, _mx_weight_sf(l1_t_sf, hidden, two_i))
    l2_t_fp8, l2_t_sf = _cast_weight_along_rows(down_proj.contiguous())
    l2_t = (l2_t_fp8, _mx_weight_sf(l2_t_sf, inter, hidden))
    return MegaMoeFP8ExpertWeights(l1=l1, l2=None, l1_t=l1_t, l2_t=l2_t)


def mega_moe_forward_fp8(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeFP8ExpertWeights,
    buffer,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    import deep_gemm

    num_tokens, hidden = x.shape
    x_fp8, x_sf = _mx_cast(x, along_rows=False)
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(_pack_ue8m0_k_major(x_sf))
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    deep_gemm.fp8_fp4_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


def mega_moe_backward_fp8(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeFP8ExpertWeights,
    buffer,
    dw_dtype: torch.dtype,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)
    l1_w, _ = weights.l1
    l2_w, _ = weights.l2_t
    dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    dw1 = torch.empty(l1_w.shape, dtype=dw_dtype, device=x.device)
    dw2 = torch.empty(l2_w.shape, dtype=dw_dtype, device=x.device)
    dtopk = torch.empty((num_tokens, buffer.num_topk), dtype=torch.float32, device=x.device)
    deep_gemm.fp8_mega_moe_backward(
        dx,
        dw1,
        dw2,
        dtopk,
        dy,
        x,
        weights.l1,
        weights.l1_t,
        weights.l2_t,
        buffer,
        activation_clamp=activation_clamp,
    )
    return dx, dw1, dw2, dtopk


def mega_moe_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    deep_gemm.bf16_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


def _bf16_weights(gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> MegaMoeExpertWeights:
    return MegaMoeExpertWeights(
        l1=gate_up_proj.to(torch.bfloat16).contiguous(), l2=down_proj.to(torch.bfloat16).contiguous()
    )


def _dw_dtype(weight: torch.Tensor) -> torch.dtype:
    return weight.dtype if weight.dtype in (torch.bfloat16, torch.float32) else torch.float32


@torch.library.custom_op("prime_rl::mega_moe_forward", mutates_args=())
def mega_moe_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    buffer = _BUFFER_REGISTRY[buffer_key]
    x_bf16 = x.to(torch.bfloat16).contiguous()
    if buffer.mma_type == "fp8xfp8":
        y = mega_moe_forward_fp8(
            x_bf16,
            topk_idx,
            topk_weights,
            _fp8_weights(gate_up_proj, down_proj, for_backward=False),
            buffer,
            activation_clamp,
        )
    else:
        y = mega_moe_forward(
            x_bf16, topk_idx, topk_weights, _bf16_weights(gate_up_proj, down_proj), buffer, activation_clamp
        )
    return y.to(x.dtype)


@mega_moe_forward_op.register_fake
def _mega_moe_forward_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::mega_moe_backward", mutates_args=())
def mega_moe_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    buffer = _BUFFER_REGISTRY[buffer_key]
    dy_bf16, x_bf16 = dy.to(torch.bfloat16).contiguous(), x.to(torch.bfloat16).contiguous()
    if buffer.mma_type == "fp8xfp8":
        dx, dw1, dw2, dtopk = mega_moe_backward_fp8(
            dy_bf16,
            x_bf16,
            topk_idx,
            topk_weights,
            _fp8_weights(gate_up_proj, down_proj, for_backward=True),
            buffer,
            _dw_dtype(gate_up_proj),
            activation_clamp=activation_clamp,
        )
    else:
        dx, dw1, dw2, dtopk = mega_moe_backward(
            dy_bf16,
            x_bf16,
            topk_idx,
            topk_weights,
            _bf16_weights(gate_up_proj, down_proj),
            buffer,
            _dw_dtype(gate_up_proj),
            activation_clamp=activation_clamp,
        )
    return dx.to(x.dtype), dw1.to(gate_up_proj.dtype), dw2.to(down_proj.dtype), dtopk


@mega_moe_backward_op.register_fake
def _mega_moe_backward_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x),
        torch.empty_like(gate_up_proj),
        torch.empty_like(down_proj),
        torch.empty_like(topk_weights, dtype=torch.float32),
    )


def _mega_moe_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer_key, activation_clamp = inputs
    ctx.save_for_backward(x, topk_idx, topk_weights, gate_up_proj, down_proj)
    ctx.buffer_key = buffer_key
    ctx.activation_clamp = activation_clamp


def _mega_moe_backward(ctx, grad_y: torch.Tensor):
    x, topk_idx, topk_weights, gate_up_proj, down_proj = ctx.saved_tensors
    dx, dw1, dw2, dtopk = torch.ops.prime_rl.mega_moe_backward(
        grad_y, x, topk_idx, topk_weights, gate_up_proj, down_proj, ctx.buffer_key, ctx.activation_clamp
    )
    return dx, None, dtopk, dw1, dw2, None, None


mega_moe_forward_op.register_autograd(_mega_moe_backward, setup_context=_mega_moe_setup_context)


def mega_moe_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
    dw_dtype: torch.dtype,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    dw1 = torch.empty(weights.l1.shape, dtype=dw_dtype, device=x.device)
    dw2 = torch.empty(weights.l2.shape, dtype=dw_dtype, device=x.device)
    dtopk = torch.empty((num_tokens, buffer.num_topk), dtype=torch.float32, device=x.device)
    deep_gemm.bf16_mega_moe_backward(
        dx,
        dw1,
        dw2,
        dtopk,
        dy,
        weights.l1,
        weights.l2,
        buffer,
        activation_clamp=activation_clamp,
    )
    return dx, dw1, dw2, dtopk


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _activation_clamp(activation) -> float | None:
    from prime_rl.trainer.models.deepseek_v4.moe import ClampedSwiglu

    if activation is Silu:
        return None
    if isinstance(activation, ClampedSwiglu):
        return float(activation.limit)
    raise ValueError("Mega MoE requires a SwiGLU (`silu` or DeepSeek V4 clamped) expert activation.")


class MegaMoEExpertCompute:
    def __init__(
        self,
        experts: "GroupedExperts",
        num_experts: int,
        top_k: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_reserved_sms: int = 16,
        precision: MegaMoePrecision = "bf16",
    ) -> None:
        if not mega_moe_available(precision):
            kernels = " and ".join(f"`{name}`" for name in _MEGA_MOE_KERNELS[precision])
            raise RuntimeError(
                "Mega MoE requires an SM100+/Blackwell GPU and the prime-mega-moe `deep_gemm` wheel from "
                f"Prime Intellect installed over the public one; this build does not expose {kernels}."
            )
        self.validate(experts)
        hidden = experts.down_proj.shape[1]
        reserve_sms_for_comm(num_reserved_sms)

        self.precision = precision
        self.activation_clamp = _activation_clamp(experts.activation)
        self.max_tokens_per_rank = max_tokens_per_rank
        self.buffer = build_mega_moe_buffer(
            group, num_experts, max_tokens_per_rank, top_k, hidden, experts.hidden_dim, precision
        )
        # Registered once here: a registry write inside the compiled, checkpointed block is a side effect dynamo rejects.
        self.buffer_key = register_mega_moe_buffer(self.buffer)

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("Mega MoE requires gated experts (SwiGLU gate+up), got non-gated experts.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("Mega MoE does not support expert biases.")
        _activation_clamp(experts.activation)
        check_mega_moe_dims(experts.down_proj.shape[1], experts.hidden_dim)

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE buffer is sized for {self.max_tokens_per_rank} tokens/rank, got {num_tokens}. "
                "Raise `model.moe.dispatch.max_tokens_per_rank`."
            )
        if experts.gate_up_proj is not None:
            gate_up_proj = _to_local(experts.gate_up_proj)
        else:
            gate_up_proj = torch.cat([_to_local(experts.gate_proj), _to_local(experts.up_proj)], dim=1)
        return torch.ops.prime_rl.mega_moe_forward(
            x,
            selected_experts_indices.to(torch.int64),
            top_scores.to(torch.float32),
            gate_up_proj,
            _to_local(experts.down_proj),
            self.buffer_key,
            self.activation_clamp,
        )
