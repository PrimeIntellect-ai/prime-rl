import torch
from fla.modules.causal_conv1d.ops import causal_conv1d_bwd, causal_conv1d_fwd
from fla.modules.norm.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd
from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid_bwd, fused_beta_sigmoid_fwd
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.kda.chunk_bwd import chunk_kda_bwd
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.utils.index import prepare_chunk_indices as fla_prepare_chunk_indices
from torch import Tensor

# FLA's public entry points are torch.compiler.disable'd (directly or through its backend dispatch).
# Inside an activation-checkpointed block that graph break makes Dynamo run the whole block eagerly,
# so each op is exposed as an opaque custom op that calls FLA's forward and backward functions.

# FLA's default chunk size; prepare_chunk_indices and every op below must agree on it.
FLA_CHUNK_SIZE = 64


def prepare_chunk_indices(cu_seqlens: Tensor) -> Tensor:
    """Each chunk's (sequence, chunk within sequence) index, which these ops take; build it once per packed row
    outside the compiled layers, since it reads ``cu_seqlens`` on the host."""
    return fla_prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)


@torch.library.custom_op("prime_rl::fla_causal_conv1d", mutates_args=())
def _causal_conv1d(
    x: Tensor, weight: Tensor, activation: str | None, cu_seqlens: Tensor, chunk_indices: Tensor
) -> Tensor:
    y, _ = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=None,
        residual=None,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return y


@_causal_conv1d.register_fake
def _(x, weight, activation, cu_seqlens, chunk_indices):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::fla_causal_conv1d_backward", mutates_args=())
def _causal_conv1d_backward(
    dy: Tensor, x: Tensor, weight: Tensor, activation: str | None, cu_seqlens: Tensor, chunk_indices: Tensor
) -> tuple[Tensor, Tensor]:
    dx, dweight, _, _, _ = causal_conv1d_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight,
        bias=None,
        residual=None,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return dx, dweight


@_causal_conv1d_backward.register_fake
def _(dy, x, weight, activation, cu_seqlens, chunk_indices):
    return torch.empty_like(x), torch.empty_like(weight)


def _causal_conv1d_setup_context(ctx, inputs, output) -> None:
    x, weight, activation, cu_seqlens, chunk_indices = inputs
    ctx.save_for_backward(x, weight, cu_seqlens, chunk_indices)
    ctx.activation = activation


def _causal_conv1d_autograd(ctx, dy: Tensor):
    x, weight, cu_seqlens, chunk_indices = ctx.saved_tensors
    dx, dweight = _causal_conv1d_backward(dy.contiguous(), x, weight, ctx.activation, cu_seqlens, chunk_indices)
    return dx, dweight, None, None, None


_causal_conv1d.register_autograd(_causal_conv1d_autograd, setup_context=_causal_conv1d_setup_context)


def causal_conv1d(
    x: Tensor, weight: Tensor, activation: str | None, cu_seqlens: Tensor, chunk_indices: Tensor
) -> Tensor:
    """FLA's varlen causal depthwise convolution without bias or states: ``x`` is (1, tokens, channels), ``weight`` is
    (channels, width)."""
    return _causal_conv1d(x.contiguous(), weight.contiguous(), activation, cu_seqlens, chunk_indices)


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule", mutates_args=())
def _chunk_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, scale: float, cu_seqlens: Tensor, chunk_indices: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    g_cumsum, o, A, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return o.to(q.dtype), q, q_rstd, k, k_rstd, g_cumsum, A


@_chunk_gated_delta_rule.register_fake
def _(q, k, v, g, beta, scale, cu_seqlens, chunk_indices):
    return (
        torch.empty_like(v),
        torch.empty_like(q),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
        torch.empty_like(k),
        k.new_empty(k.shape[:-1], dtype=torch.float32),
        torch.empty_like(g, dtype=torch.float32),
        k.new_empty((*beta.shape, FLA_CHUNK_SIZE)),
    )


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule_backward", mutates_args=())
def _chunk_gated_delta_rule_backward(
    do: Tensor,
    q: Tensor,
    q_rstd: Tensor,
    k: Tensor,
    k_rstd: Tensor,
    v: Tensor,
    g_cumsum: Tensor,
    beta: Tensor,
    A: Tensor,
    scale: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbeta, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=None,
        do=do,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    dq = l2norm_bwd(q, q_rstd, dq)
    dk = l2norm_bwd(k, k_rstd, dk)
    return dq.to(q), dk.to(k), dv.to(v), dg.to(g_cumsum), dbeta.to(beta)


@_chunk_gated_delta_rule_backward.register_fake
def _(do, q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, scale, cu_seqlens, chunk_indices):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g_cumsum),
        torch.empty_like(beta),
    )


def _chunk_gated_delta_rule_setup_context(ctx, inputs, output) -> None:
    _, _, v, g, beta, scale, cu_seqlens, chunk_indices = inputs
    _, q, q_rstd, k, k_rstd, g_cumsum, A = output
    ctx.mark_non_differentiable(q, q_rstd, k, k_rstd, g_cumsum, A)
    ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, cu_seqlens, chunk_indices)
    ctx.scale = scale
    ctx.g_dtype = g.dtype


def _chunk_gated_delta_rule_autograd(ctx, do: Tensor, *_):
    q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, cu_seqlens, chunk_indices = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = _chunk_gated_delta_rule_backward(
        do.contiguous(), q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, ctx.scale, cu_seqlens, chunk_indices
    )
    return dq, dk, dv, dg.to(ctx.g_dtype), dbeta, None, None, None


_chunk_gated_delta_rule.register_autograd(
    _chunk_gated_delta_rule_autograd, setup_context=_chunk_gated_delta_rule_setup_context
)


def chunk_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor, chunk_indices: Tensor
) -> Tensor:
    """FLA's varlen ``chunk_gated_delta_rule`` with in-kernel q/k l2norm and no states: q, k and v are
    (1, tokens, heads, head_dim), g (log decay) and beta are (1, tokens, heads)."""
    return _chunk_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        k.shape[-1] ** -0.5,
        cu_seqlens,
        chunk_indices,
    )[0]


@torch.library.custom_op("prime_rl::fla_chunk_kda", mutates_args=())
def _chunk_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    beta_sigmoid = fused_beta_sigmoid_fwd(beta)
    o, _, _, Aqk, Akk, *_ = chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta_sigmoid,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    return o.type_as(q), q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk


@_chunk_kda.register_fake
def _(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices):
    return (
        torch.empty_like(v),
        torch.empty_like(q),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
        torch.empty_like(k),
        k.new_empty(k.shape[:-1], dtype=torch.float32),
        torch.empty_like(beta, dtype=torch.float32),
        k.new_empty((*beta.shape, FLA_CHUNK_SIZE)),
        k.new_empty((*beta.shape, FLA_CHUNK_SIZE)),
    )


@torch.library.custom_op("prime_rl::fla_chunk_kda_backward", mutates_args=())
def _chunk_kda_backward(
    do: Tensor,
    q: Tensor,
    q_rstd: Tensor,
    k: Tensor,
    k_rstd: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    beta_sigmoid: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    Aqk: Tensor,
    Akk: Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbeta_sigmoid, dg, _, dA_log, ddt_bias = chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        beta=beta_sigmoid,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=None,
        do=do,
        dht=None,
        g_org=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    dq = l2norm_bwd(q, q_rstd, dq)
    dk = l2norm_bwd(k, k_rstd, dk)
    dbeta = fused_beta_sigmoid_bwd(x=beta, dy=dbeta_sigmoid)
    return dq.to(q), dk.to(k), dv.to(v), dg.to(g), dbeta.to(beta), dA_log.to(A_log), ddt_bias.to(dt_bias)


@_chunk_kda_backward.register_fake
def _(
    do,
    q,
    q_rstd,
    k,
    k_rstd,
    v,
    g,
    beta,
    beta_sigmoid,
    A_log,
    dt_bias,
    Aqk,
    Akk,
    scale,
    lower_bound,
    cu_seqlens,
    chunk_indices,
):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        torch.empty_like(A_log),
        torch.empty_like(dt_bias),
    )


def _chunk_kda_setup_context(ctx, inputs, output) -> None:
    _, _, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices = inputs
    _, q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk = output
    ctx.mark_non_differentiable(q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk)
    ctx.save_for_backward(
        q, q_rstd, k, k_rstd, v, g, beta, beta_sigmoid, A_log, dt_bias, Aqk, Akk, cu_seqlens, chunk_indices
    )
    ctx.scale = scale
    ctx.lower_bound = lower_bound


def _chunk_kda_autograd(ctx, do: Tensor, *_):
    q, q_rstd, k, k_rstd, v, g, beta, beta_sigmoid, A_log, dt_bias, Aqk, Akk, cu_seqlens, chunk_indices = (
        ctx.saved_tensors
    )
    grads = _chunk_kda_backward(
        do.contiguous(),
        q,
        q_rstd,
        k,
        k_rstd,
        v,
        g,
        beta,
        beta_sigmoid,
        A_log,
        dt_bias,
        Aqk,
        Akk,
        ctx.scale,
        ctx.lower_bound,
        cu_seqlens,
        chunk_indices,
    )
    return *grads, None, None, None, None


_chunk_kda.register_autograd(_chunk_kda_autograd, setup_context=_chunk_kda_setup_context)


def chunk_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    lower_bound: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
) -> Tensor:
    """FLA's varlen ``chunk_kda`` with in-kernel q/k l2norm, safe gate and beta sigmoid, and no states: q, k, v and g
    (raw gate) are (1, tokens, heads, head_dim), beta (logits) is (1, tokens, heads)."""
    return _chunk_kda(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        k.shape[-1] ** -0.5,
        lower_bound,
        cu_seqlens,
        chunk_indices,
    )[0]


@torch.library.custom_op("prime_rl::fla_rms_norm_gated", mutates_args=())
def _rms_norm_gated(x: Tensor, gate: Tensor, weight: Tensor, eps: float, activation: str) -> tuple[Tensor, Tensor]:
    y, _, rstd, _ = layer_norm_gated_fwd(
        x=x, g=gate, weight=weight, bias=None, activation=activation, eps=eps, is_rms_norm=True
    )
    return y, rstd


@_rms_norm_gated.register_fake
def _(x, gate, weight, eps, activation):
    return torch.empty_like(x), x.new_empty(x.shape[0], dtype=torch.float32)


@torch.library.custom_op("prime_rl::fla_rms_norm_gated_backward", mutates_args=())
def _rms_norm_gated_backward(
    dy: Tensor, x: Tensor, gate: Tensor, weight: Tensor, rstd: Tensor, eps: float, activation: str
) -> tuple[Tensor, Tensor, Tensor]:
    dx, dgate, dweight, _, _ = layer_norm_gated_bwd(
        dy=dy,
        x=x,
        g=gate,
        weight=weight,
        bias=None,
        activation=activation,
        eps=eps,
        rstd=rstd,
        is_rms_norm=True,
        x_dtype=x.dtype,
    )
    return dx, dgate, dweight


@_rms_norm_gated_backward.register_fake
def _(dy, x, gate, weight, rstd, eps, activation):
    return torch.empty_like(x), torch.empty_like(gate), torch.empty_like(weight)


def _rms_norm_gated_setup_context(ctx, inputs, output) -> None:
    x, gate, weight, eps, activation = inputs
    _, rstd = output
    ctx.mark_non_differentiable(rstd)
    ctx.save_for_backward(x, gate, weight, rstd)
    ctx.eps = eps
    ctx.activation = activation


def _rms_norm_gated_autograd(ctx, dy: Tensor, _drstd: Tensor | None):
    x, gate, weight, rstd = ctx.saved_tensors
    dx, dgate, dweight = _rms_norm_gated_backward(dy.contiguous(), x, gate, weight, rstd, ctx.eps, ctx.activation)
    return dx, dgate, dweight, None, None


_rms_norm_gated.register_autograd(_rms_norm_gated_autograd, setup_context=_rms_norm_gated_setup_context)


def rms_norm_gated(x: Tensor, gate: Tensor, weight: Tensor, eps: float, activation: str) -> Tensor:
    """FLA's gated RMSNorm (``FusedRMSNormGated``) without residual or bias: x and gate are (tokens, hidden)."""
    return _rms_norm_gated(x.contiguous(), gate.contiguous(), weight.contiguous(), eps, activation)[0]
