from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torchtitan.distributed.expert_parallel import expert_parallel

try:
    from prime_rl.trainer.models.layers.moe_triton import (
        triton_histogram,
        triton_index_compute,
        triton_index_scatter,
        triton_moe_scatter,
        triton_moe_weighted_gather,
    )
except ImportError:
    triton_histogram = None
    triton_index_compute = None
    triton_index_scatter = None
    triton_moe_scatter = None
    triton_moe_weighted_gather = None

RoutingBackend = Literal["torch", "triton"]
ScatterBackend = Literal["torch", "triton"]
GatherBackend = Literal["torch", "triton"]
GroupedGemmBackend = Literal["loop", "torch_grouped_mm"]
RoutedFFNBackend = Literal["torch", "fused"]


@dataclass(frozen=True)
class MoEBackendSettings:
    routing: RoutingBackend = "torch"
    scatter: ScatterBackend = "torch"
    gather: GatherBackend = "torch"
    grouped_gemm: GroupedGemmBackend = "torch_grouped_mm"
    routed_ffn: RoutedFFNBackend = "torch"


def resolve_moe_backend_settings(
    *,
    use_grouped_mm: bool,
    routing: str,
    scatter: str,
    gather: str,
    routed_ffn: str,
) -> MoEBackendSettings:
    return MoEBackendSettings(
        routing=routing,  # type: ignore[arg-type]
        scatter=scatter,  # type: ignore[arg-type]
        gather=gather,  # type: ignore[arg-type]
        grouped_gemm="torch_grouped_mm" if use_grouped_mm else "loop",
        routed_ffn=routed_ffn,  # type: ignore[arg-type]
    )


def _require_triton(backend_name: str, impl) -> None:
    if impl is None:
        raise RuntimeError(f"{backend_name} backend requires Triton and CUDA support")


def histogram(top_k_rank: torch.Tensor, expert_num: int, backend: RoutingBackend) -> torch.Tensor:
    if backend == "triton":
        _require_triton("MoE routing", triton_histogram)
        return triton_histogram(top_k_rank, expert_num)
    return _torch_histogram(top_k_rank, expert_num)


def _torch_histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    if top_k_rank.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_k_rank must be int32 or int64 tensor")
    if expert_num < 0:
        raise ValueError("expert_num must be non-negative")
    if expert_num == 0:
        return torch.zeros((0,), device=top_k_rank.device, dtype=torch.int32)
    if top_k_rank.numel() == 0:
        return torch.zeros((expert_num,), device=top_k_rank.device, dtype=torch.int32)

    flat = top_k_rank.reshape(-1)
    valid = flat >= 0
    if valid.numel() > 0:
        valid_any = valid.any()
        max_val = flat.masked_fill(~valid, -1).amax()
        if not (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()):
            if bool(valid_any) and bool(max_val >= expert_num):
                raise ValueError("top_k_rank contains out-of-range expert index")
        torch._assert(
            torch.logical_or(~valid_any, max_val < expert_num),
            "top_k_rank contains out-of-range expert index",
        )

    counts = torch.bincount(flat[valid].to(torch.int64), minlength=expert_num)
    return counts.to(torch.int32)


def index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor, backend: RoutingBackend) -> torch.Tensor:
    if backend == "triton":
        _require_triton("MoE routing", triton_index_compute)
        return triton_index_compute(indices, expert_histogram)
    return _torch_index_compute(indices, expert_histogram)


def _torch_index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64 tensor")
    if expert_histogram.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_histogram must be int32 or int64 tensor")
    if indices.dim() != 2:
        raise ValueError("indices must be 2D [token_num, top_k]")
    if expert_histogram.dim() != 1:
        raise ValueError("expert_histogram must be 1D [num_experts]")
    if indices.device != expert_histogram.device:
        raise ValueError("indices and expert_histogram must be on the same device")

    num_experts = expert_histogram.numel()
    total_num = indices.numel()
    out = indices.to(torch.int32, copy=True)
    if total_num == 0:
        return out

    flat = indices.reshape(-1)
    valid = flat >= 0
    if valid.numel() > 0:
        valid_any = valid.any()
        max_val = flat.masked_fill(~valid, -1).amax()
        torch._assert(
            torch.logical_or(~valid_any, max_val < num_experts),
            "indices contains out-of-range expert index",
        )

    torch._assert(~(expert_histogram < 0).any(), "expert_histogram must be non-negative")
    torch._assert(
        expert_histogram.sum() == valid.sum(),
        "expert_histogram does not match number of valid indices",
    )
    if num_experts == 0:
        torch._assert(~valid.any(), "num_experts must be positive when indices are valid")

    hist = expert_histogram.to(torch.int64)
    base_offset = torch.cumsum(hist, dim=0) - hist

    positions = torch.arange(total_num, device=indices.device, dtype=torch.int64)
    valid_pos = positions[valid]
    valid_expert = flat[valid].to(torch.int64)

    sort_key = valid_expert * total_num + valid_pos
    order = torch.argsort(sort_key)
    sorted_pos = valid_pos[order]
    sorted_expert = valid_expert[order]

    idx = torch.arange(sorted_expert.numel(), device=indices.device, dtype=torch.int64)
    group_change = torch.ones_like(sorted_expert, dtype=torch.bool)
    group_change[1:] = sorted_expert[1:] != sorted_expert[:-1]
    group_start = torch.zeros_like(sorted_expert, dtype=torch.int64)
    group_start[group_change] = idx[group_change]
    group_start = torch.cummax(group_start, dim=0).values
    within = idx - group_start

    scatter_pos_sorted = base_offset[sorted_expert] + within
    flat_out = out.reshape(-1).to(torch.int64)
    flat_out[sorted_pos] = scatter_pos_sorted
    return flat_out.reshape_as(out).to(torch.int32)


class _MoEWeightedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("weight must be a floating tensor")
        if index.dim() != 2 or weight.dim() != 2:
            raise ValueError("index and weight must be 2D [token_num, top_k]")
        if weight.shape != index.shape:
            raise ValueError("index and weight must have the same shape")

        token_num, top_k = index.shape
        hidden_dim = input.shape[-1]

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        safe_index = flat_index.clamp_min(0).to(torch.int64)
        gathered = input.index_select(0, safe_index)
        gathered = gathered * valid.to(gathered.dtype).unsqueeze(-1)
        acc_dtype = torch.float32 if weight.dtype == torch.float32 else input.dtype
        if gathered.dtype != acc_dtype:
            gathered = gathered.to(acc_dtype)
        weight_flat = weight.reshape(-1, 1).to(acc_dtype)
        out = (gathered * weight_flat).reshape(token_num, top_k, hidden_dim).sum(dim=1)
        if out.dtype != input.dtype:
            out = out.to(input.dtype)

        ctx.save_for_backward(input, index, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        input, index, weight = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = input.shape[-1]

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        safe_index = flat_index.clamp_min(0).to(torch.int64)
        gathered = input.index_select(0, safe_index).reshape(token_num, top_k, hidden_dim)
        gathered = gathered * valid.reshape(token_num, top_k, 1).to(gathered.dtype)

        grad_weight = (grad_out.unsqueeze(1) * gathered).sum(dim=2)
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)

        grad_in = input.new_zeros(input.shape)
        grad_contrib = (grad_out.unsqueeze(1) * weight.to(grad_out.dtype).unsqueeze(-1)).reshape(-1, hidden_dim)
        grad_contrib = grad_contrib * valid.to(grad_contrib.dtype).unsqueeze(-1)
        grad_in.index_add_(0, safe_index, grad_contrib.to(grad_in.dtype))
        return grad_in, None, grad_weight


def moe_weighted_gather(
    input: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    backend: GatherBackend,
) -> torch.Tensor:
    if backend == "triton":
        _require_triton("MoE gather", triton_moe_weighted_gather)
        return triton_moe_weighted_gather(input, index, weight)
    return _MoEWeightedGather.apply(input, index, weight)


class _MoEScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if index.dim() != 2:
            raise ValueError("index must be 2D [token_num, top_k]")
        if input.dim() != 2:
            raise ValueError("input must be 2D [token_num, hidden_dim]")

        token_num, top_k = index.shape
        if input.shape[0] != token_num:
            raise ValueError("input and index must have the same token_num")

        hidden_dim = input.shape[-1]
        out_size = (index >= 0).sum()
        flat_index = index.reshape(-1)

        out = input.new_zeros((out_size, hidden_dim))
        if out_size == 0 or flat_index.numel() == 0:
            ctx.save_for_backward(index)
            ctx.hidden_dim = hidden_dim
            return out

        valid = flat_index >= 0
        flat_input = input.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
        if valid.all():
            out.index_copy_(0, flat_index.to(torch.int64), flat_input)
        else:
            flat_index = flat_index[valid].to(torch.int64)
            flat_input = flat_input[valid]
            if flat_index.numel() > 0:
                out.index_copy_(0, flat_index, flat_input)

        ctx.save_for_backward(index)
        ctx.hidden_dim = hidden_dim
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = ctx.hidden_dim

        if grad_out.numel() == 0:
            return grad_out.new_zeros((token_num, hidden_dim)), None

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        if valid.all():
            gathered = grad_out.index_select(0, flat_index.to(torch.int64))
            grad_input = gathered.reshape(token_num, top_k, hidden_dim).sum(dim=1)
        else:
            grad_input = grad_out.new_zeros((token_num, hidden_dim))
            if valid.any():
                gathered = grad_out.index_select(0, flat_index[valid].to(torch.int64))
                tmp = grad_out.new_zeros((flat_index.numel(), hidden_dim))
                tmp[valid] = gathered
                grad_input = tmp.reshape(token_num, top_k, hidden_dim).sum(dim=1)
        return grad_input, None


def moe_scatter(input: torch.Tensor, index: torch.Tensor, backend: ScatterBackend) -> torch.Tensor:
    if backend == "triton":
        _require_triton("MoE scatter", triton_moe_scatter)
        return triton_moe_scatter(input, index)
    return _MoEScatter.apply(input, index)


def _scatter_tokens_by_expert_with_histogram(
    input: torch.Tensor,
    expert_indices: torch.Tensor,
    experts_histogram: torch.Tensor,
    backends: MoEBackendSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backends.routed_ffn == "fused":
        _require_triton("Fused routed MoE path", triton_index_scatter)
        return triton_index_scatter(input, expert_indices, experts_histogram)

    scatter_index = index_compute(expert_indices, experts_histogram, backends.routing)
    scattered = moe_scatter(input, scatter_index, backends.scatter)
    return scattered, scatter_index


def scatter_tokens_by_expert(
    input: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    backends: MoEBackendSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts_histogram = histogram(expert_indices, num_experts, backends.routing)
    scattered, scatter_index = _scatter_tokens_by_expert_with_histogram(
        input,
        expert_indices,
        experts_histogram,
        backends,
    )
    return scattered, scatter_index, experts_histogram


def combine_expert_outputs(
    expert_output: torch.Tensor,
    scatter_index: torch.Tensor,
    token_weights: torch.Tensor,
    backend: GatherBackend,
) -> torch.Tensor:
    return moe_weighted_gather(expert_output, scatter_index, token_weights, backend)


def scatter_token_weights(token_weights: torch.Tensor, scatter_index: torch.Tensor) -> torch.Tensor:
    expert_weights = token_weights.new_zeros((max(int((scatter_index >= 0).sum().item()), 0),))
    if expert_weights.numel() == 0:
        return expert_weights
    valid = scatter_index >= 0
    expert_weights[scatter_index[valid].to(torch.int64)] = token_weights[valid]
    return expert_weights


def run_routed_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    input: torch.Tensor,
    expert_indices: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    num_experts: int,
    score_before_experts: bool,
    backends: MoEBackendSettings,
) -> torch.Tensor:
    experts_histogram = histogram(expert_indices, num_experts, backends.routing)
    routed_input, scatter_index = _scatter_tokens_by_expert_with_histogram(
        input,
        expert_indices,
        experts_histogram,
        backends,
    )

    if not score_before_experts:
        routed_output = run_grouped_experts(
            w1,
            w2,
            w3,
            routed_input,
            experts_histogram,
            backends.grouped_gemm,
        )
        return combine_expert_outputs(routed_output, scatter_index, token_weights, backends.gather)

    expert_weights = scatter_token_weights(token_weights, scatter_index)
    routed_input = (routed_input.to(torch.float32) * expert_weights.reshape(-1, 1)).to(input.dtype)
    routed_output = run_grouped_experts(
        w1,
        w2,
        w3,
        routed_input,
        experts_histogram,
        backends.grouped_gemm,
    )
    combine_weights = torch.ones_like(token_weights)
    return combine_expert_outputs(routed_output, scatter_index, combine_weights, backends.gather)


@expert_parallel
def run_experts_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    num_padding = x.shape[0] - sum(num_tokens_per_expert_list)

    x = torch.split(
        x[: sum(num_tokens_per_expert_list)],
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


@expert_parallel
def run_experts_torch_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


def run_grouped_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    backend: GroupedGemmBackend,
) -> torch.Tensor:
    if backend == "loop":
        return run_experts_loop(w1, w2, w3, x, num_tokens_per_expert)
    return run_experts_torch_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
