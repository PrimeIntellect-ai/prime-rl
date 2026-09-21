from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup


def mega_moe_available() -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_capability() >= (10, 0)
        and hasattr(deep_gemm, "bf16_mega_moe")
        and hasattr(deep_gemm, "bf16_mega_moe_backward")
    )


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


def prepare_mega_moe_weights(gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> MegaMoeExpertWeights:
    import deep_gemm

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        gate_up_proj.to(torch.bfloat16).contiguous(), down_proj.to(torch.bfloat16).contiguous()
    )
    return MegaMoeExpertWeights(l1=l1, l2=l2)


def reserve_sms_for_comm(num_reserved_sms: int) -> None:
    import deep_gemm

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    deep_gemm.set_num_sms(max(total - num_reserved_sms, 1))


_BUFFER_CACHE: dict[tuple, object] = {}


def build_mega_moe_buffer(
    group: ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate_hidden: int,
):
    import deep_gemm

    check_mega_moe_dims(hidden, intermediate_hidden)
    key = (id(group), num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden)
    buffer = _BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden, mma_type="bf16xbf16"
        )
        _BUFFER_CACHE[key] = buffer
    return buffer


def _stage_inputs(buffer, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    buffer.x[:num_tokens].copy_(x)
    buffer.topk_idx[:num_tokens].copy_(topk_idx.view(num_tokens, buffer.num_topk))
    buffer.topk_weights[:num_tokens].copy_(topk_weights.view(num_tokens, buffer.num_topk))


def mega_moe_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
) -> torch.Tensor:
    """Fused dispatch + SwiGLU MLP + combine for this rank's raw (pre-dispatch) bf16 tokens ``x``.
    Router weights are applied at combine time. Returns bf16 ``(num_tokens, hidden)``."""
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    deep_gemm.bf16_mega_moe(y, weights.l1, weights.l2, buffer)
    return y


def mega_moe_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
    dw_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    dw1 = torch.empty(weights.l1.shape, dtype=dw_dtype, device=x.device)
    dw2 = torch.empty(weights.l2.shape, dtype=dw_dtype, device=x.device)
    dtopk = torch.empty((num_tokens, buffer.num_topk), dtype=torch.float32, device=x.device)
    deep_gemm.bf16_mega_moe_backward(dx, dw1, dw2, dtopk, dy, weights.l1, weights.l2, buffer, dw_natural_layout=True)
    return dx, dw1, dw2, dtopk
