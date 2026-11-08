"""Fused hyper-connection read (`HyperConnection.mix`) for Qwen3.8 Flash Next: Triton kernels behind custom ops.

Per token, with `S` streams of width `D`:

    n = rms_norm_per_stream(x) * (1 + norm_weight)
    a = silu((n @ down_weight.T) / S)
    block_input = mean_s(sigmoid(a @ up_weight.T)_s * n_s)
    injection = 2 * sigmoid((n @ inject_weight.T) / S)

Here the read is `mix`, the write is `combine` (streams + block_output * injection) and the write gates are the
injection weights. One GEMM projects the unnormalized streams with the gain folded into `[down; inject]`, in fp32;
each stream's rstd scales its rows after it. `n` and the gate logits are never stored: the mix kernel recomputes the
logits per tile and normalizes in registers. The forward saves only the streams, the per-stream rstd and the fp32
projection; the backward recomputes the rest. A read can first apply the previous sublayer's write in the pass that
computes the rstd, so a write and the next read share it.
"""

import triton
import triton.language as tl


@triton.jit
def _rstd_kernel(
    streams_ptr,
    block_output_ptr,
    inject_ptr,
    new_streams_ptr,
    rstd_ptr,
    M,
    D,
    eps,
    NS: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One stream's rstd for a row block. With WRITE, the stream is first updated with the previous sublayer's write,
    streams + block_output * inject, and stored to new_streams."""
    E = NS * D
    s = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    sum_squares = tl.zeros([BLOCK_M], dtype=tl.float32)
    if WRITE:
        inject = tl.load(inject_ptr + rows * NS + s, mask=row_mask, other=0.0).to(tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (offs_d < D)[None, :]
        offsets = rows[:, None] * E + s * D + offs_d[None, :]
        x = tl.load(streams_ptr + offsets, mask=mask, other=0.0)
        if WRITE:
            block_output = tl.load(block_output_ptr + rows[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
            x = (x.to(tl.float32) + block_output.to(tl.float32) * inject[:, None]).to(new_streams_ptr.dtype.element_ty)
            tl.store(new_streams_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        sum_squares += tl.sum(x * x, axis=1)
    tl.store(rstd_ptr + rows * NS + s, tl.rsqrt(sum_squares / D + eps), mask=row_mask)


@triton.jit
def _gates_kernel(
    own_ptr,
    rstd_ptr,
    projected_ptr,
    hidden_ptr,
    inject_ptr,
    M,
    R,
    P_PAD,
    NS: tl.constexpr,
    NI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """projected = the streams' projections scaled by their rstd and averaged; hidden = silu of its first R columns;
    inject = 2 sigmoid of the rest."""
    P = R + NI
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    offs_p = tl.arange(0, BLOCK_P)
    mask = row_mask[:, None] & (offs_p < P)[None, :]
    projected = tl.zeros([BLOCK_M, BLOCK_P], dtype=tl.float32)
    for s in tl.static_range(NS):
        own = tl.load(own_ptr + (s * M + rows[:, None]) * P_PAD + offs_p[None, :], mask=mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows * NS + s, mask=row_mask, other=0.0)
        projected += own * rstd[:, None]
    projected = projected / NS
    tl.store(projected_ptr + rows[:, None] * P + offs_p[None, :], projected, mask=mask)
    gate = tl.sigmoid(projected)
    hidden = (projected * gate).to(hidden_ptr.dtype.element_ty)
    tl.store(hidden_ptr + rows[:, None] * R + offs_p[None, :], hidden, mask=mask & (offs_p < R)[None, :])
    inject = (2.0 * gate).to(inject_ptr.dtype.element_ty)
    tl.store(inject_ptr + rows[:, None] * NI + offs_p[None, :] - R, inject, mask=mask & (offs_p >= R)[None, :])


@triton.jit
def _sigmoid(x):
    """sigmoid(x) = (1 + tanh(x / 2)) / 2, with the hardware tanh: one special-function instruction."""
    tanh = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [x * 0.5], dtype=tl.float32, is_pure=True, pack=1
    )
    return 0.5 + 0.5 * tanh


@triton.jit
def _hidden_tile(hidden_ptr, rows, row_mask, r0, R, BLOCK_R: tl.constexpr):
    """hidden[rows, r0:r0 + BLOCK_R]."""
    offs_r = r0 + tl.arange(0, BLOCK_R)
    return tl.load(
        hidden_ptr + rows[:, None] * R + offs_r[None, :], mask=row_mask[:, None] & (offs_r < R)[None, :], other=0.0
    )


@triton.jit
def _up_tile(up_weight_ptr, cols, col_mask, r0, R, BLOCK_R: tl.constexpr):
    """up_weight[cols, r0:r0 + BLOCK_R].T"""
    offs_r = r0 + tl.arange(0, BLOCK_R)
    return tl.load(
        up_weight_ptr + cols[None, :] * R + offs_r[:, None], mask=col_mask[None, :] & (offs_r < R)[:, None], other=0.0
    )


@triton.jit
def _gate_logits(
    hidden_ptr,
    up_weight_ptr,
    rows,
    row_mask,
    cols,
    col_mask,
    R,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """hidden @ up_weight.T for a row block and the channels ``cols``: the gate logits, over BLOCK_R ranks per step."""
    logits = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for r0 in range(0, R, BLOCK_R):
        hidden = _hidden_tile(hidden_ptr, rows, row_mask, r0, R, BLOCK_R)
        logits = tl.dot(hidden, _up_tile(up_weight_ptr, cols, col_mask, r0, R, BLOCK_R), logits)
    return logits


@triton.jit
def _mix_kernel(
    streams_ptr,
    norm_weight_ptr,
    up_weight_ptr,
    rstd_ptr,
    hidden_ptr,
    out_ptr,
    M,
    D,
    R,
    NS: tl.constexpr,
    ONE_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """out = mean over the streams of sigmoid(hidden @ up_weight.T) * normalized streams. With ONE_STEP (the rank
    fits one BLOCK_R step) the hidden tile stays in registers across the streams."""
    E = NS * D
    offs_d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    mask = row_mask[:, None] & d_mask[None, :]
    if ONE_STEP:
        hidden = _hidden_tile(hidden_ptr, rows, row_mask, 0, R, BLOCK_R)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for s in range(NS):
        cols = s * D + offs_d
        if ONE_STEP:
            logits = tl.dot(hidden, _up_tile(up_weight_ptr, cols, d_mask, 0, R, BLOCK_R))
        else:
            logits = _gate_logits(hidden_ptr, up_weight_ptr, rows, row_mask, cols, d_mask, R, BLOCK_M, BLOCK_D, BLOCK_R)
        x = tl.load(streams_ptr + rows[:, None] * E + cols[None, :], mask=mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows * NS + s, mask=row_mask, other=0.0)
        gain = 1.0 + tl.load(norm_weight_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
        acc += _sigmoid(logits) * (x.to(tl.float32) * rstd[:, None] * gain[None, :])
    tl.store(out_ptr + rows[:, None] * D + offs_d[None, :], (acc / NS).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gate_backward_kernel(
    streams_ptr,
    norm_weight_ptr,
    up_weight_ptr,
    rstd_ptr,
    hidden_ptr,
    grad_out_ptr,
    grad_logits_ptr,
    row_dot_partial_ptr,
    M,
    D,
    R,
    NS: tl.constexpr,
    ONE_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """For BLOCK_D channels of one stream and a share of the rows: the gradient of the gate logits, grad_out / NS *
    normalized * gate * (1 - gate), and the channel block's share of the stream's row dot product of the norm
    backward through the mean, sum(grad_out / NS * gate * normalized). With ONE_STEP the up-projection tile stays in
    registers across the rows."""
    E = NS * D
    d_blocks = tl.cdiv(D, BLOCK_D)
    s = tl.program_id(0) // d_blocks
    offs_d = (tl.program_id(0) % d_blocks) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    cols = s * D + offs_d
    gain = 1.0 + tl.load(norm_weight_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    if ONE_STEP:
        w = _up_tile(up_weight_ptr, cols, d_mask, 0, R, BLOCK_R)
    for m0 in range(tl.program_id(1) * BLOCK_M, M, tl.num_programs(1) * BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask = row_mask[:, None] & d_mask[None, :]
        if ONE_STEP:
            logits = tl.dot(_hidden_tile(hidden_ptr, rows, row_mask, 0, R, BLOCK_R), w)
        else:
            logits = _gate_logits(hidden_ptr, up_weight_ptr, rows, row_mask, cols, d_mask, R, BLOCK_M, BLOCK_D, BLOCK_R)
        gate = _sigmoid(logits)
        offsets = rows[:, None] * E + cols[None, :]
        x = tl.load(streams_ptr + offsets, mask=mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows * NS + s, mask=row_mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + rows[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        grad_gated = grad_out.to(tl.float32) / NS * x.to(tl.float32) * rstd[:, None] * gain[None, :]
        tl.store(
            row_dot_partial_ptr + ((tl.program_id(0) % d_blocks) * M + rows) * NS + s,
            tl.sum(grad_gated * gate, axis=1),
            mask=row_mask,
        )
        grad_logits = (grad_gated * gate * (1.0 - gate)).to(grad_logits_ptr.dtype.element_ty)
        tl.store(grad_logits_ptr + offsets, grad_logits, mask=mask)


@triton.jit
def _projection_backward_kernel(
    projected_ptr,
    grad_hidden_ptr,
    grad_inject_ptr,
    own_ptr,
    rstd_ptr,
    row_dot_ptr,
    grad_projected_ptr,
    scaled_ptr,
    M,
    R,
    P_PAD,
    NS: tl.constexpr,
    NI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """The gradient of the projection (through silu for the gate input and 2 sigmoid for the write gates, / NS for
    its scale; rows padded with zeros to P_PAD), each stream's copy scaled by its rstd (for the down projection's
    weight gradient), and grad_projected . (each stream's own projection, scaled by its rstd) added to the row dot
    products."""
    P = R + NI
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    offs_p = tl.arange(0, BLOCK_P)
    mask = row_mask[:, None] & (offs_p < P)[None, :]
    pad_mask = row_mask[:, None] & (offs_p < P_PAD)[None, :]
    z = tl.load(projected_ptr + rows[:, None] * P + offs_p[None, :], mask=mask, other=0.0)
    sig = tl.sigmoid(z)
    hidden_mask = row_mask[:, None] & (offs_p < R)[None, :]
    inject_mask = row_mask[:, None] & (offs_p >= R)[None, :] & (offs_p < P)[None, :]
    grad = tl.load(grad_hidden_ptr + rows[:, None] * R + offs_p[None, :], mask=hidden_mask, other=0.0).to(tl.float32)
    grad = tl.where(offs_p[None, :] < R, grad * sig * (1.0 + z * (1.0 - sig)), 0.0)
    grad_inject = tl.load(grad_inject_ptr + rows[:, None] * NI + offs_p[None, :] - R, mask=inject_mask, other=0.0)
    grad = (grad + grad_inject.to(tl.float32) * 2.0 * sig * (1.0 - sig)) / NS
    out = rows[:, None] * P_PAD + offs_p[None, :]
    tl.store(grad_projected_ptr + out, grad.to(grad_projected_ptr.dtype.element_ty), mask=pad_mask)
    for s in tl.static_range(NS):
        own = tl.load(own_ptr + s * M * P_PAD + out, mask=mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows * NS + s, mask=row_mask, other=0.0)
        row_dot = tl.load(row_dot_ptr + rows * NS + s, mask=row_mask, other=0.0) + rstd * tl.sum(grad * own, axis=1)
        tl.store(row_dot_ptr + rows * NS + s, row_dot, mask=row_mask)
        scaled = (grad * rstd[:, None]).to(scaled_ptr.dtype.element_ty)
        tl.store(scaled_ptr + s * M * P_PAD + out, scaled, mask=pad_mask)


@triton.jit
def _stream_backward_kernel(
    streams_ptr,
    norm_weight_ptr,
    up_weight_ptr,
    rstd_ptr,
    hidden_ptr,
    grad_out_ptr,
    grad_projection_ptr,
    row_dot_ptr,
    grad_streams_in_ptr,
    grad_streams_ptr,
    grad_norm_partial_ptr,
    M,
    D,
    R,
    NS: tl.constexpr,
    HAS_GRAD_STREAMS_IN: tl.constexpr,
    ONE_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """For BLOCK_D channels of one stream and a share of the rows: the stream gradient, the norm backward of
    grad_normalized = grad_out / NS * gate + grad_projection (the projection's gradient through down_weight), plus
    the gradient the streams get elsewhere (grad_streams_in), and the partial sums of the norm weight's gradient. With
    ONE_STEP the up-projection tile stays in registers across the rows."""
    E = NS * D
    d_blocks = tl.cdiv(D, BLOCK_D)
    s = tl.program_id(0) // d_blocks
    offs_d = (tl.program_id(0) % d_blocks) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    cols = s * D + offs_d
    gain = 1.0 + tl.load(norm_weight_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    if ONE_STEP:
        w = _up_tile(up_weight_ptr, cols, d_mask, 0, R, BLOCK_R)
    grad_gain = tl.zeros([BLOCK_D], dtype=tl.float32)
    for m0 in range(tl.program_id(1) * BLOCK_M, M, tl.num_programs(1) * BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask = row_mask[:, None] & d_mask[None, :]
        if ONE_STEP:
            logits = tl.dot(_hidden_tile(hidden_ptr, rows, row_mask, 0, R, BLOCK_R), w)
        else:
            logits = _gate_logits(hidden_ptr, up_weight_ptr, rows, row_mask, cols, d_mask, R, BLOCK_M, BLOCK_D, BLOCK_R)
        gate = _sigmoid(logits)
        offsets = rows[:, None] * E + cols[None, :]
        grad_out = tl.load(grad_out_ptr + rows[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        grad_normalized = grad_out.to(tl.float32) / NS * gate
        grad_normalized += tl.load(grad_projection_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows * NS + s, mask=row_mask, other=0.0)
        x_hat = tl.load(streams_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * rstd[:, None]
        grad_gain += tl.sum(grad_normalized * x_hat, axis=0)
        row_dot = tl.load(row_dot_ptr + rows * NS + s, mask=row_mask, other=0.0)
        grad_x = rstd[:, None] * (grad_normalized * gain[None, :] - x_hat * (row_dot / D)[:, None])
        if HAS_GRAD_STREAMS_IN:
            grad_x += tl.load(grad_streams_in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_streams_ptr + offsets, grad_x.to(grad_streams_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_norm_partial_ptr + tl.program_id(1) * E + cols, grad_gain, mask=d_mask)


@triton.jit
def _write_backward_kernel(
    grad_streams_ptr,
    block_output_ptr,
    inject_ptr,
    grad_block_output_ptr,
    grad_inject_ptr,
    M,
    D,
    NS: tl.constexpr,
    NS_POW2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """The gradients of a write, streams + block_output * inject: grad_block_output = sum over the streams of
    inject * grad_streams, grad_inject = each stream's sum of grad_streams * block_output."""
    E = NS * D
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    offs_s = tl.arange(0, NS_POW2)
    grad_inject = tl.zeros([BLOCK_M, NS_POW2], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (offs_d < D)[None, :]
        block_output = tl.load(block_output_ptr + rows[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        grad_block_output = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        for s in tl.static_range(NS):
            grad = tl.load(grad_streams_ptr + rows[:, None] * E + s * D + offs_d[None, :], mask=mask, other=0.0)
            grad = grad.to(tl.float32)
            inject = tl.load(inject_ptr + rows * NS + s, mask=row_mask, other=0.0).to(tl.float32)
            grad_block_output += grad * inject[:, None]
            stream_sum = tl.sum(grad * block_output.to(tl.float32), axis=1)
            grad_inject += tl.where(offs_s[None, :] == s, stream_sum[:, None], 0.0)
        tl.store(
            grad_block_output_ptr + rows[:, None] * D + offs_d[None, :],
            grad_block_output.to(grad_block_output_ptr.dtype.element_ty),
            mask=mask,
        )
    tl.store(
        grad_inject_ptr + rows[:, None] * NS + offs_s[None, :],
        grad_inject.to(grad_inject_ptr.dtype.element_ty),
        mask=row_mask[:, None] & (offs_s < NS)[None, :],
    )
