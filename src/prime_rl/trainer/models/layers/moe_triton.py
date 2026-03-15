# Adapted in part from StepTronOSS Triton MoE kernels:
# - `SteptronOss/steptronoss/model/optimizations/moe_routing/triton.py`
# - `SteptronOss/steptronoss/model/optimizations/moe_gather/triton.py`
# - `SteptronOss/steptronoss/model/optimizations/moe_scatter/triton.py`
# - `SteptronOss/steptronoss/model/optimizations/routed_grouped_ffn/triton.py`
from __future__ import annotations

import torch
import triton
import triton.language as tl

_GRAD_WEIGHT_CHUNK_BYTES = 512 * 1024 * 1024


def _triton_dtype(tensor: torch.Tensor):
    if tensor.dtype == torch.float16:
        return tl.float16
    if tensor.dtype == torch.bfloat16:
        return tl.bfloat16
    if tensor.dtype == torch.float32:
        return tl.float32
    return None


def _compute_grad_weight_chunked(input: torch.Tensor, index: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    token_num, top_k = index.shape
    if token_num == 0 or top_k == 0:
        return grad_out.new_zeros((token_num, top_k))

    hidden_dim = input.shape[-1]
    compute_dtype = torch.promote_types(input.dtype, grad_out.dtype)
    grad_out_row = grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)
    grad_out_row = grad_out_row.unsqueeze(1)

    valid = index >= 0
    safe_index = index.clamp_min(0).to(torch.int64)
    grad_weight = grad_out.new_empty((token_num, top_k), dtype=compute_dtype)

    bytes_per_topk = token_num * hidden_dim * max(
        input.element_size(),
        torch.empty((), dtype=compute_dtype).element_size(),
    )
    chunk_k = max(1, min(top_k, _GRAD_WEIGHT_CHUNK_BYTES // max(1, bytes_per_topk)))

    for start in range(0, top_k, chunk_k):
        end = min(start + chunk_k, top_k)
        gathered = input.index_select(0, safe_index[:, start:end].reshape(-1)).reshape(
            token_num, end - start, hidden_dim
        )
        if gathered.dtype != compute_dtype:
            gathered = gathered.to(compute_dtype)
        gathered.mul_(grad_out_row)
        grad_chunk = gathered.sum(dim=2)
        grad_chunk.masked_fill_(~valid[:, start:end], 0)
        grad_weight[:, start:end] = grad_chunk

    return grad_weight


@triton.jit
def _histogram_kernel(
    index_ptr,
    out_ptr,
    n_elements,
    expert_num,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements
    idx = tl.load(index_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (idx >= 0) & (idx < expert_num)
    tl.atomic_add(out_ptr + idx, 1, mask=valid)


@triton.jit
def _count_per_block_kernel(
    index_ptr,
    count_ptr,
    n_elements,
    expert_num,
    stride_count0,
    stride_count1,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_expert = tl.program_id(1)

    offs_n = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_elements
    idx = tl.load(index_ptr + offs_n, mask=mask_n, other=-1).to(tl.int32)

    offs_e = pid_expert * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_e = offs_e < expert_num

    match = idx[:, None] == offs_e[None, :]
    valid = mask_n[:, None] & mask_e[None, :] & (idx[:, None] >= 0)
    counts = tl.sum((match & valid).to(tl.int32), axis=0)

    out_ptrs = count_ptr + pid_block * stride_count0 + offs_e * stride_count1
    tl.store(out_ptrs, counts, mask=mask_e)


@triton.jit
def _index_compute_kernel(
    index_ptr,
    base_offset_ptr,
    block_offset_ptr,
    out_ptr,
    n_elements,
    expert_num,
    stride_block0,
    stride_block1,
    BLOCK_N: tl.constexpr,
):
    pid_block = tl.program_id(0)
    offs = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements

    idx = tl.load(index_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (idx >= 0) & (idx < expert_num)
    safe_idx = tl.where(valid, idx, 0)

    positions = tl.arange(0, BLOCK_N)
    same_expert = safe_idx[:, None] == safe_idx[None, :]
    earlier = positions[None, :] < positions[:, None]
    valid_prev = valid[None, :]
    local_rank = tl.sum((same_expert & earlier & valid_prev).to(tl.int32), axis=1)

    base_offset = tl.load(base_offset_ptr + safe_idx, mask=valid, other=0)
    block_ptrs = block_offset_ptr + pid_block * stride_block0 + safe_idx * stride_block1
    block_offset = tl.load(block_ptrs, mask=valid, other=0)
    out = base_offset + block_offset + local_rank
    tl.store(out_ptr + offs, out, mask=valid)


@triton.jit
def _index_scatter_kernel(
    input_ptr,
    index_ptr,
    base_offset_ptr,
    block_offset_ptr,
    out_index_ptr,
    out_ptr,
    n_elements,
    expert_num,
    top_k,
    hidden_dim,
    stride_in0,
    stride_in1,
    stride_block0,
    stride_block1,
    stride_out0,
    stride_out1,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    H_BLOCKS: tl.constexpr,
):
    pid_block = tl.program_id(0)
    offs = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements

    idx = tl.load(index_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (idx >= 0) & (idx < expert_num)
    safe_idx = tl.where(valid, idx, 0)

    positions = tl.arange(0, BLOCK_N)
    same_expert = safe_idx[:, None] == safe_idx[None, :]
    earlier = positions[None, :] < positions[:, None]
    valid_prev = valid[None, :]
    local_rank = tl.sum((same_expert & earlier & valid_prev).to(tl.int32), axis=1)

    base_offset = tl.load(base_offset_ptr + safe_idx, mask=valid, other=0)
    block_ptrs = block_offset_ptr + pid_block * stride_block0 + safe_idx * stride_block1
    block_offset = tl.load(block_ptrs, mask=valid, other=0)
    scatter_pos = base_offset + block_offset + local_rank

    tl.store(out_index_ptr + offs, tl.where(valid, scatter_pos, idx), mask=mask)

    token_idx = offs // top_k
    for h_block in range(H_BLOCKS):
        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < hidden_dim
        input_ptrs = input_ptr + token_idx[:, None] * stride_in0 + offs_h[None, :] * stride_in1
        vals = tl.load(input_ptrs, mask=valid[:, None] & mask_h[None, :], other=0.0)
        out_ptrs = out_ptr + scatter_pos[:, None] * stride_out0 + offs_h[None, :] * stride_out1
        tl.store(out_ptrs, vals, mask=valid[:, None] & mask_h[None, :])


@triton.jit
def _index_scatter_backward_kernel(
    grad_out_ptr,
    scatter_index_ptr,
    grad_input_ptr,
    token_num,
    top_k,
    hidden_dim,
    stride_go0,
    stride_go1,
    stride_idx0,
    stride_idx1,
    stride_gi0,
    stride_gi1,
    OUT_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= token_num:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for k in range(0, top_k):
        idx = tl.load(scatter_index_ptr + pid_t * stride_idx0 + k * stride_idx1).to(tl.int32)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0)

        grad_ptrs = grad_out_ptr + safe_idx * stride_go0 + offs_h * stride_go1
        vals = tl.load(grad_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.where(valid, vals, 0.0)

    out = acc.to(OUT_DTYPE)
    out_ptrs = grad_input_ptr + pid_t * stride_gi0 + offs_h * stride_gi1
    tl.store(out_ptrs, out, mask=mask_h)


@triton.jit
def _moe_weighted_gather_kernel(
    in_ptr,
    index_ptr,
    weight_ptr,
    out_ptr,
    token_num,
    top_k,
    hidden_dim,
    stride_in0,
    stride_in1,
    stride_idx0,
    stride_idx1,
    stride_w0,
    stride_w1,
    stride_out0,
    stride_out1,
    ACC_FP32: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= token_num:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim

    acc_dtype = tl.float32 if ACC_FP32 else IN_DTYPE
    acc = tl.zeros((BLOCK_H,), dtype=acc_dtype)

    for k in range(0, top_k):
        idx = tl.load(index_ptr + pid_t * stride_idx0 + k * stride_idx1).to(tl.int32)
        valid = idx >= 0
        idx = tl.where(valid, idx, 0)

        weight = tl.load(weight_ptr + pid_t * stride_w0 + k * stride_w1).to(acc_dtype)
        weight = tl.where(valid, weight, 0)

        in_ptrs = in_ptr + idx * stride_in0 + offs_h * stride_in1
        vals = tl.load(in_ptrs, mask=mask_h, other=0.0).to(acc_dtype)
        acc += vals * weight

    out = acc.to(IN_DTYPE)
    out_ptrs = out_ptr + pid_t * stride_out0 + offs_h * stride_out1
    tl.store(out_ptrs, out, mask=mask_h)


@triton.jit
def _moe_weighted_gather_grad_in_kernel(
    grad_out_ptr,
    index_ptr,
    weight_ptr,
    grad_in_ptr,
    token_num,
    top_k,
    hidden_dim,
    stride_go0,
    stride_go1,
    stride_idx0,
    stride_idx1,
    stride_w0,
    stride_w1,
    stride_gi0,
    stride_gi1,
    IN_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= token_num:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim
    grad_ptrs = grad_out_ptr + pid_t * stride_go0 + offs_h * stride_go1
    grad_vals = tl.load(grad_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    for k in range(0, top_k):
        idx = tl.load(index_ptr + pid_t * stride_idx0 + k * stride_idx1).to(tl.int32)
        valid = idx >= 0
        weight = tl.load(weight_ptr + pid_t * stride_w0 + k * stride_w1).to(tl.float32)
        contrib = grad_vals * tl.where(valid, weight, 0.0)
        out_ptrs = grad_in_ptr + tl.where(valid, idx, 0) * stride_gi0 + offs_h * stride_gi1
        tl.atomic_add(out_ptrs, contrib.to(IN_DTYPE), mask=valid & mask_h)


def triton_histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    if not isinstance(top_k_rank, torch.Tensor):
        raise TypeError("top_k_rank must be a torch.Tensor")
    if top_k_rank.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_k_rank must be int32 or int64 tensor")
    if expert_num < 0:
        raise ValueError("expert_num must be non-negative")
    if expert_num == 0:
        return torch.zeros((0,), device=top_k_rank.device, dtype=torch.int32)
    if top_k_rank.numel() == 0:
        return torch.zeros((expert_num,), device=top_k_rank.device, dtype=torch.int32)
    if not top_k_rank.is_cuda:
        raise ValueError("triton_histogram requires CUDA tensor inputs")

    flat = top_k_rank.reshape(-1)
    valid = flat >= 0
    valid_any = valid.any()
    max_val = flat.masked_fill(~valid, -1).amax()
    torch._assert(
        torch.logical_or(~valid_any, max_val < expert_num),
        "top_k_rank contains out-of-range expert index",
    )

    flat_i32 = flat.to(torch.int32).contiguous()
    counts = torch.zeros((expert_num,), device=flat.device, dtype=torch.int32)
    _histogram_kernel[(triton.cdiv(flat_i32.numel(), 256),)](
        flat_i32,
        counts,
        flat_i32.numel(),
        expert_num,
        BLOCK_N=256,
    )
    return counts


def triton_index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if not isinstance(expert_histogram, torch.Tensor):
        raise TypeError("expert_histogram must be a torch.Tensor")
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
    if not indices.is_cuda:
        raise ValueError("triton_index_compute requires CUDA tensor inputs")

    num_experts = expert_histogram.numel()
    total_num = indices.numel()
    out = indices.to(torch.int32, copy=True)
    if total_num == 0:
        return out

    flat = indices.reshape(-1)
    valid = flat >= 0
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

    flat_i32 = flat.to(torch.int32).contiguous()
    out_flat = out.reshape(-1).contiguous()
    hist_i32 = expert_histogram.to(torch.int32)
    base_offset = (torch.cumsum(hist_i32, dim=0) - hist_i32).contiguous()

    block_n = 128
    block_e = 32
    num_blocks = triton.cdiv(total_num, block_n)
    count = torch.empty((num_blocks, num_experts), device=indices.device, dtype=torch.int32)
    _count_per_block_kernel[(num_blocks, triton.cdiv(num_experts, block_e))](
        flat_i32,
        count,
        total_num,
        num_experts,
        count.stride(0),
        count.stride(1),
        BLOCK_N=block_n,
        BLOCK_E=block_e,
    )
    block_offset = (torch.cumsum(count, dim=0) - count).contiguous()

    _index_compute_kernel[(num_blocks,)](
        flat_i32,
        base_offset,
        block_offset,
        out_flat,
        total_num,
        num_experts,
        block_offset.stride(0),
        block_offset.stride(1),
        BLOCK_N=block_n,
    )
    return out_flat.reshape_as(out)


class _TritonIndexScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor, expert_histogram: torch.Tensor):
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if not isinstance(expert_histogram, torch.Tensor):
            raise TypeError("expert_histogram must be a torch.Tensor")
        if input.dim() != 2:
            raise ValueError("input must be 2D [token_num, hidden_dim]")
        if indices.dim() != 2:
            raise ValueError("indices must be 2D [token_num, top_k]")
        if expert_histogram.dim() != 1:
            raise ValueError("expert_histogram must be 1D [num_experts]")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("indices must be int32 or int64 tensor")
        if expert_histogram.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_histogram must be int32 or int64 tensor")
        if input.device != indices.device or input.device != expert_histogram.device:
            raise ValueError("input, indices and expert_histogram must be on the same device")
        if not input.is_cuda:
            raise ValueError("triton_index_scatter requires CUDA tensor inputs")
        if input.shape[0] != indices.shape[0]:
            raise ValueError("input and indices must have the same token_num")

        token_num, top_k = indices.shape
        hidden_dim = input.shape[-1]
        num_experts = expert_histogram.numel()
        out_size = int(expert_histogram.sum().item())

        out = input.new_zeros((out_size, hidden_dim))
        out_index = indices.to(torch.int32, copy=True)
        if out_size == 0 or indices.numel() == 0:
            ctx.save_for_backward(out_index)
            ctx.hidden_dim = hidden_dim
            return out, out_index

        flat = indices.reshape(-1)
        valid = flat >= 0
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

        hist_i32 = expert_histogram.to(torch.int32)
        base_offset = (torch.cumsum(hist_i32, dim=0) - hist_i32).contiguous()
        flat_i32 = flat.to(torch.int32).contiguous()
        out_index_flat = out_index.reshape(-1).contiguous()

        block_n = 128
        block_e = 32
        block_h = 128
        num_blocks = triton.cdiv(flat_i32.numel(), block_n)
        count = torch.empty((num_blocks, num_experts), device=indices.device, dtype=torch.int32)
        _count_per_block_kernel[(num_blocks, triton.cdiv(num_experts, block_e))](
            flat_i32,
            count,
            flat_i32.numel(),
            num_experts,
            count.stride(0),
            count.stride(1),
            BLOCK_N=block_n,
            BLOCK_E=block_e,
        )
        block_offset = (torch.cumsum(count, dim=0) - count).contiguous()

        _index_scatter_kernel[(num_blocks,)](
            input.contiguous(),
            flat_i32,
            base_offset,
            block_offset,
            out_index_flat,
            out,
            flat_i32.numel(),
            num_experts,
            top_k,
            hidden_dim,
            input.stride(0),
            input.stride(1),
            block_offset.stride(0),
            block_offset.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            H_BLOCKS=triton.cdiv(hidden_dim, block_h),
        )

        ctx.save_for_backward(out_index)
        ctx.hidden_dim = hidden_dim
        return out, out_index

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_index: torch.Tensor | None):
        del grad_index
        (scatter_index,) = ctx.saved_tensors
        token_num, top_k = scatter_index.shape
        hidden_dim = ctx.hidden_dim

        if grad_out.numel() == 0:
            return grad_out.new_zeros((token_num, hidden_dim)), None, None

        grad_out = grad_out.contiguous()
        grad_input = grad_out.new_empty((token_num, hidden_dim))
        block_h = 128
        out_dtype = _triton_dtype(grad_out)
        if out_dtype is None:
            raise TypeError("triton_index_scatter backward only supports fp16/bf16/fp32")

        _index_scatter_backward_kernel[(token_num, triton.cdiv(hidden_dim, block_h))](
            grad_out,
            scatter_index,
            grad_input,
            token_num,
            top_k,
            hidden_dim,
            grad_out.stride(0),
            grad_out.stride(1),
            scatter_index.stride(0),
            scatter_index.stride(1),
            grad_input.stride(0),
            grad_input.stride(1),
            OUT_DTYPE=out_dtype,
            BLOCK_H=block_h,
        )
        return grad_input, None, None


class _TritonMoEWeightedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("weight must be a floating tensor")
        if index.dim() != 2 or weight.dim() != 2:
            raise ValueError("index and weight must be 2D [token_num, top_k]")

        token_num, top_k = index.shape
        if weight.shape != index.shape:
            raise ValueError("index and weight must have the same shape")
        hidden_dim = input.shape[-1]

        out = input.new_zeros((token_num, hidden_dim))
        if token_num == 0 or index.numel() == 0:
            ctx.save_for_backward(input, index, weight)
            return out

        in_dtype = _triton_dtype(input)
        if in_dtype is None:
            raise TypeError("input must be fp16/bf16/fp32")

        block_h = 128
        _moe_weighted_gather_kernel[(token_num, triton.cdiv(hidden_dim, block_h))](
            input,
            index,
            weight,
            out,
            token_num,
            top_k,
            hidden_dim,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            ACC_FP32=weight.dtype == torch.float32,
            IN_DTYPE=in_dtype,
            BLOCK_H=block_h,
        )

        ctx.save_for_backward(input, index, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        input, index, weight = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = input.shape[-1]

        grad_weight = _compute_grad_weight_chunked(input, index, grad_out)
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)

        grad_in = input.new_zeros(input.shape)
        in_dtype = _triton_dtype(input)
        if in_dtype is None:
            raise TypeError("input must be fp16/bf16/fp32")

        block_h = 128
        _moe_weighted_gather_grad_in_kernel[(token_num, triton.cdiv(hidden_dim, block_h))](
            grad_out.contiguous(),
            index,
            weight,
            grad_in,
            token_num,
            top_k,
            hidden_dim,
            grad_out.stride(0),
            grad_out.stride(1),
            index.stride(0),
            index.stride(1),
            weight.stride(0),
            weight.stride(1),
            grad_in.stride(0),
            grad_in.stride(1),
            IN_DTYPE=in_dtype,
            BLOCK_H=block_h,
        )
        return grad_in, None, grad_weight


def triton_index_scatter(
    input: torch.Tensor,
    indices: torch.Tensor,
    expert_histogram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TritonIndexScatter.apply(input, indices, expert_histogram)


def triton_moe_weighted_gather(input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _TritonMoEWeightedGather.apply(input, index, weight)
