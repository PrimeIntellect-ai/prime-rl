"""Fused hyper-connection read (`HyperConnection.mix`) for Qwen3.8 Flash Next.

Per token, with `S` streams of width `D`:

    n = rms_norm_per_stream(x) * (1 + norm_weight)
    a = silu((n @ down_weight.T) / S)
    block_input = mean_s(sigmoid(a @ up_weight.T)_s * n_s)
    injection = 2 * sigmoid((n @ inject_weight.T) / S)

Autograd keeps only `x`, the per-stream rstd and the `(T, low_rank + S)` projection; backward
recomputes `n` and `a @ up_weight.T`. Both passes walk the tokens in chunks, so the `(T, S * D)`
intermediates exist for one chunk at a time.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CHUNK_TOKENS = 16384
NORM_BWD_ROWS = 32
MIX_TOKENS = 2
MIX_BLOCK = 512


@triton.jit
def _norm_fwd_kernel(x_ptr, weight_ptr, out_ptr, rstd_ptr, eps, S: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x) / D + eps)
    weight = tl.load(weight_ptr + (row % S) * D + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * D + offs, (x * rstd * (1.0 + weight)).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _mix_fwd_kernel(
    z_ptr, n_ptr, out_ptr, num_tokens, S: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows < num_tokens)[:, None] & (cols < D)[None, :]
    acc = tl.zeros([BT, BD], dtype=tl.float32)
    for s in tl.static_range(S):
        ptrs = rows[:, None] * S * D + s * D + cols[None, :]
        z = tl.load(z_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        n = tl.load(n_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sigmoid(z) * n
    tl.store(out_ptr + rows[:, None] * D + cols[None, :], (acc / S).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mix_bwd_kernel(
    z_ptr,
    n_ptr,
    grad_out_ptr,
    grad_n_ptr,
    num_tokens,
    S: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Writes the block input's gradient w.r.t. `n` to `grad_n` and overwrites the logits `z` with the gradient w.r.t. them."""
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows < num_tokens)[:, None] & (cols < D)[None, :]
    grad = tl.load(grad_out_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32) / S
    for s in tl.static_range(S):
        ptrs = rows[:, None] * S * D + s * D + cols[None, :]
        u = tl.sigmoid(tl.load(z_ptr + ptrs, mask=mask, other=0.0).to(tl.float32))
        n = tl.load(n_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_n_ptr + ptrs, (grad * u).to(grad_n_ptr.dtype.element_ty), mask=mask)
        tl.store(z_ptr + ptrs, (grad * n * u * (1.0 - u)).to(z_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _norm_bwd_kernel(
    x_ptr,
    weight_ptr,
    rstd_ptr,
    grad_n_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    num_tokens,
    S: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (ROWS tokens, stream); writes its partial weight gradient to row `pid` of `grad_weight`."""
    pid = tl.program_id(0)
    stream = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    scale = 1.0 + tl.load(weight_ptr + stream * D + offs, mask=mask, other=0.0).to(tl.float32)
    grad_weight = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(ROWS):
        token = pid.to(tl.int64) * ROWS + i
        row_mask = mask & (token < num_tokens)
        ptrs = token * S * D + stream * D + offs
        rstd = tl.load(rstd_ptr + token * S + stream, mask=token < num_tokens, other=0.0)
        x_hat = tl.load(x_ptr + ptrs, mask=row_mask, other=0.0).to(tl.float32) * rstd
        grad_n = tl.load(grad_n_ptr + ptrs, mask=row_mask, other=0.0).to(tl.float32)
        grad_weight += grad_n * x_hat
        grad_x_hat = grad_n * scale
        grad_x = rstd * (grad_x_hat - x_hat * (tl.sum(grad_x_hat * x_hat) / D))
        tl.store(grad_x_ptr + ptrs, grad_x.to(grad_x_ptr.dtype.element_ty), mask=row_mask)
    tl.store(grad_weight_ptr + pid * S * D + stream * D + offs, grad_weight, mask=mask)


def _norm(x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor, stream_count: int, eps: float) -> torch.Tensor:
    num_tokens, expanded_size = x.shape
    hidden_size = expanded_size // stream_count
    out = torch.empty_like(x)
    _norm_fwd_kernel[(num_tokens * stream_count,)](
        x, weight, out, rstd, eps, S=stream_count, D=hidden_size, BLOCK=triton.next_power_of_2(hidden_size)
    )
    return out


def _mix(kernel, *args, num_tokens: int, stream_count: int, hidden_size: int) -> None:
    block = min(MIX_BLOCK, triton.next_power_of_2(hidden_size))
    grid = (triton.cdiv(num_tokens, MIX_TOKENS), triton.cdiv(hidden_size, block))
    kernel[grid](*args, num_tokens, S=stream_count, D=hidden_size, BT=MIX_TOKENS, BD=block, num_warps=4)


def _projection_weight(down_weight: torch.Tensor, inject_weight: torch.Tensor | None) -> torch.Tensor:
    """`[down_weight; inject_weight]`, zero-padded to a multiple of 8 rows so cuBLAS picks aligned kernels."""
    weights = [down_weight] if inject_weight is None else [down_weight, inject_weight]
    rows = sum(weight.shape[0] for weight in weights)
    padding = -rows % 8
    if padding:
        weights.append(down_weight.new_zeros(padding, down_weight.shape[1]))
    return torch.cat(weights) if len(weights) > 1 else down_weight


def _mix_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    stream_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded_size = x.shape[-1]
    hidden_size = expanded_size // stream_count
    low_rank = down_weight.shape[0]
    flat_x = x.reshape(-1, expanded_size)
    num_tokens = flat_x.shape[0]
    projection_weight = _projection_weight(down_weight, inject_weight)
    rstd = x.new_empty(num_tokens, stream_count, dtype=torch.float32)
    projection = x.new_empty(num_tokens, projection_weight.shape[0])
    block_input = x.new_empty(num_tokens, hidden_size)
    for start in range(0, num_tokens, CHUNK_TOKENS):
        chunk = slice(start, min(start + CHUNK_TOKENS, num_tokens))
        normalized = _norm(flat_x[chunk], norm_weight, rstd[chunk], stream_count, eps)
        torch.mm(normalized, projection_weight.t(), out=projection[chunk])
        mixing_logits = F.silu(projection[chunk, :low_rank] / stream_count) @ up_weight.t()
        _mix(
            _mix_fwd_kernel,
            mixing_logits,
            normalized,
            block_input[chunk],
            num_tokens=chunk.stop - start,
            stream_count=stream_count,
            hidden_size=hidden_size,
        )
    if inject_weight is None:
        injection = x.new_empty(0)
    else:
        injection = 2.0 * torch.sigmoid(projection[:, low_rank : low_rank + stream_count] / stream_count)
        injection = injection.view(*x.shape[:-1], stream_count)
    return block_input.view(*x.shape[:-1], hidden_size), injection, rstd, projection


def _mix_backward(
    grad_block_input: torch.Tensor,
    grad_injection: torch.Tensor | None,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    rstd: torch.Tensor,
    projection: torch.Tensor,
    stream_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded_size = x.shape[-1]
    hidden_size = expanded_size // stream_count
    low_rank = down_weight.shape[0]
    flat_x = x.reshape(-1, expanded_size)
    num_tokens = flat_x.shape[0]
    grad_block_input = grad_block_input.reshape(num_tokens, hidden_size)
    projection_weight = _projection_weight(down_weight, inject_weight)

    grad_x = torch.empty_like(flat_x)
    grad_norm_weight = torch.zeros(expanded_size, device=x.device, dtype=torch.float32)
    grad_projection_weight = torch.zeros(projection_weight.shape, device=x.device, dtype=torch.float32)
    grad_up_weight = torch.zeros(up_weight.shape, device=x.device, dtype=torch.float32)
    for start in range(0, num_tokens, CHUNK_TOKENS):
        chunk = slice(start, min(start + CHUNK_TOKENS, num_tokens))
        chunk_tokens = chunk.stop - start
        normalized = _norm(flat_x[chunk], norm_weight, torch.empty_like(rstd[chunk]), stream_count, eps)
        down_pre = projection[chunk, :low_rank] / stream_count
        down = F.silu(down_pre)
        grad_logits = down @ up_weight.t()
        grad_normalized = torch.empty_like(normalized)
        _mix(
            _mix_bwd_kernel,
            grad_logits,
            normalized,
            grad_block_input[chunk],
            grad_normalized,
            num_tokens=chunk_tokens,
            stream_count=stream_count,
            hidden_size=hidden_size,
        )
        grad_down = grad_logits @ up_weight
        grad_up_weight += torch.mm(grad_logits.t(), down, out_dtype=torch.float32)
        del grad_logits

        down_pre = down_pre.float()
        down_sigmoid = torch.sigmoid(down_pre)
        grad_projection = torch.zeros(chunk_tokens, projection.shape[1], device=x.device, dtype=torch.float32)
        grad_projection[:, :low_rank] = (
            grad_down * down_sigmoid * (1.0 + down_pre * (1.0 - down_sigmoid)) / stream_count
        )
        if inject_weight is not None:
            inject_sigmoid = torch.sigmoid(
                (projection[chunk, low_rank : low_rank + stream_count] / stream_count).float()
            )
            grad_inject = grad_injection.reshape(num_tokens, stream_count)[chunk]
            grad_projection[:, low_rank : low_rank + stream_count] = (
                grad_inject * 2.0 * inject_sigmoid * (1.0 - inject_sigmoid) / stream_count
            )
        grad_projection = grad_projection.to(x.dtype)
        grad_normalized.addmm_(grad_projection, projection_weight)
        grad_projection_weight += torch.mm(grad_projection.t(), normalized, out_dtype=torch.float32)
        del normalized

        partial_grad_norm_weight = x.new_empty(
            triton.cdiv(chunk_tokens, NORM_BWD_ROWS), expanded_size, dtype=torch.float32
        )
        _norm_bwd_kernel[(partial_grad_norm_weight.shape[0], stream_count)](
            flat_x[chunk],
            norm_weight,
            rstd[chunk],
            grad_normalized,
            grad_x[chunk],
            partial_grad_norm_weight,
            chunk_tokens,
            S=stream_count,
            D=hidden_size,
            ROWS=NORM_BWD_ROWS,
            BLOCK=triton.next_power_of_2(hidden_size),
        )
        grad_norm_weight += partial_grad_norm_weight.sum(0)

    inject_rows = 0 if inject_weight is None else inject_weight.shape[0]
    return (
        grad_x.view_as(x),
        grad_norm_weight.to(norm_weight.dtype),
        grad_projection_weight[:low_rank].to(down_weight.dtype),
        grad_up_weight.to(up_weight.dtype),
        grad_projection_weight[low_rank : low_rank + inject_rows].to(x.dtype),
    )


@torch.library.custom_op("prime_rl::qwen3_8_hc_mix", mutates_args=())
def _hc_mix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    stream_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _mix_forward(x, norm_weight, down_weight, up_weight, inject_weight, stream_count, eps)


@_hc_mix.register_fake
def _hc_mix_fake(x, norm_weight, down_weight, up_weight, inject_weight, stream_count, eps):
    num_tokens = x.numel() // x.shape[-1]
    projection_size = down_weight.shape[0] + (0 if inject_weight is None else inject_weight.shape[0])
    projection_size += -projection_size % 8
    injection = x.new_empty(0) if inject_weight is None else x.new_empty(*x.shape[:-1], stream_count)
    return (
        x.new_empty(*x.shape[:-1], x.shape[-1] // stream_count),
        injection,
        x.new_empty(num_tokens, stream_count, dtype=torch.float32),
        x.new_empty(num_tokens, projection_size),
    )


@torch.library.custom_op("prime_rl::qwen3_8_hc_mix_backward", mutates_args=())
def _hc_mix_backward(
    grad_block_input: torch.Tensor,
    grad_injection: torch.Tensor | None,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    rstd: torch.Tensor,
    projection: torch.Tensor,
    stream_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _mix_backward(
        grad_block_input,
        grad_injection,
        x,
        norm_weight,
        down_weight,
        up_weight,
        inject_weight,
        rstd,
        projection,
        stream_count,
        eps,
    )


@_hc_mix_backward.register_fake
def _hc_mix_backward_fake(
    grad_block_input,
    grad_injection,
    x,
    norm_weight,
    down_weight,
    up_weight,
    inject_weight,
    rstd,
    projection,
    stream_count,
    eps,
):
    inject_rows = 0 if inject_weight is None else inject_weight.shape[0]
    return (
        torch.empty_like(x),
        torch.empty_like(norm_weight),
        torch.empty_like(down_weight),
        torch.empty_like(up_weight),
        x.new_empty(inject_rows, x.shape[-1]),
    )


def _hc_mix_setup_context(ctx, inputs, output) -> None:
    x, norm_weight, down_weight, up_weight, inject_weight, stream_count, eps = inputs
    _, _, rstd, projection = output
    ctx.mark_non_differentiable(rstd, projection)
    ctx.save_for_backward(x, norm_weight, down_weight, up_weight, inject_weight, rstd, projection)
    ctx.stream_count = stream_count
    ctx.eps = eps


def _hc_mix_autograd_backward(ctx, grad_block_input, grad_injection, _grad_rstd, _grad_projection):
    x, norm_weight, down_weight, up_weight, inject_weight, rstd, projection = ctx.saved_tensors
    grad_x, grad_norm_weight, grad_down_weight, grad_up_weight, grad_inject_weight = _hc_mix_backward(
        grad_block_input.contiguous(),
        None if inject_weight is None else grad_injection.contiguous(),
        x,
        norm_weight,
        down_weight,
        up_weight,
        inject_weight,
        rstd,
        projection,
        ctx.stream_count,
        ctx.eps,
    )
    return (
        grad_x,
        grad_norm_weight,
        grad_down_weight,
        grad_up_weight,
        None if inject_weight is None else grad_inject_weight,
        None,
        None,
    )


_hc_mix.register_autograd(_hc_mix_autograd_backward, setup_context=_hc_mix_setup_context)


def fused_hc_mix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    stream_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns `(block_input, injection)`; `injection` is None without `inject_weight`."""
    block_input, injection, _, _ = _hc_mix(
        x.contiguous(), norm_weight, down_weight, up_weight, inject_weight, stream_count, eps
    )
    return block_input, None if inject_weight is None else injection
