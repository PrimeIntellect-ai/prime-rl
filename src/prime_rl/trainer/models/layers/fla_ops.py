from dataclasses import dataclass

import torch
import torch.distributed as dist
from fla.modules.causal_conv1d.cp import CausalConv1dFunctionCP
from fla.modules.causal_conv1d.ops import causal_conv1d_bwd, causal_conv1d_fwd
from fla.modules.norm.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd
from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid_bwd, fused_beta_sigmoid_fwd
from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.kda.chunk_bwd import chunk_kda_bwd
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.utils.index import prepare_chunk_indices as fla_prepare_chunk_indices
from torch import Tensor

from prime_rl.utils.cp import CPContext

# FLA's public entry points are torch.compiler.disable'd (directly or through its backend dispatch).
# Inside an activation-checkpointed block that graph break makes Dynamo run the whole block eagerly,
# so each op is exposed as an opaque custom op that calls FLA's forward and backward functions.

# FLA's default chunk size; prepare_chunk_indices and every op below must agree on it.
FLA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class FlaVarlen:
    """This rank's documents of a packed row, as the ops below take them."""

    local_cu_seqlens: Tensor
    """`[local_documents + 1]` boundaries of this rank's documents; the whole row's without CP."""
    chunk_indices: Tensor
    """`[chunks, 2]` each chunk's (document, chunk within document) index."""
    cp_boundary: Tensor | None = None
    """CPU int64 `[5]`: FLA's is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks and pre_num_conv_tokens, as a
    tensor so that every CP rank compiles the same graph; None without CP."""


def prepare_chunk_indices(cu_seqlens: Tensor) -> Tensor:
    """Each chunk's (sequence, chunk within sequence) index, which these ops take; build it once per packed row
    outside the compiled layers, since it reads ``cu_seqlens`` on the host."""
    return fla_prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)


def prepare_fla_varlen(cu_seqlens: Tensor, cp_context: CPContext) -> FlaVarlen:
    """Build a packed row's ``FlaVarlen`` from its global ``cu_seqlens`` once, outside the compiled layers, since it
    reads ``cu_seqlens`` on the host."""
    if not cp_context.cp_enabled:
        return FlaVarlen(cu_seqlens, prepare_chunk_indices(cu_seqlens))
    fla_cp_context = build_cp_context(cu_seqlens, group=cp_context.cp_group)
    cp_boundary = torch.tensor(
        [
            fla_cp_context.is_first_rank,
            fla_cp_context.is_last_rank,
            fla_cp_context.pre_num_ranks,
            fla_cp_context.post_num_ranks,
            fla_cp_context.pre_num_conv_tokens,
        ],
        dtype=torch.int64,
    )
    chunk_indices = fla_prepare_chunk_indices(
        fla_cp_context.cu_seqlens, FLA_CHUNK_SIZE, cu_seqlens_cpu=fla_cp_context.cu_seqlens_cpu
    )
    return FlaVarlen(fla_cp_context.cu_seqlens, chunk_indices, cp_boundary)


def _fla_cp_context(cu_seqlens: Tensor, cp_boundary: Tensor, group_name: str) -> FLACPContext:
    is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks, pre_num_conv_tokens = cp_boundary.tolist()
    return FLACPContext(
        group=dist.distributed_c10d._resolve_process_group(group_name),
        cu_seqlens=cu_seqlens,
        is_first_rank=bool(is_first_rank),
        is_last_rank=bool(is_last_rank),
        pre_num_ranks=pre_num_ranks,
        post_num_ranks=post_num_ranks,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )


def _h0_fake(k: Tensor, v: Tensor) -> Tensor:
    return k.new_empty((1, v.shape[2], k.shape[-1], v.shape[-1]), dtype=torch.float32)


def _conv_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation: str | None,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    initial_state: Tensor | None,
) -> Tensor:
    y, _ = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return y


def _conv_bwd(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation: str | None,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    dx, dweight, dbias, _, dh0 = causal_conv1d_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return dx, dweight, dbias, dh0


@torch.library.custom_op("prime_rl::fla_causal_conv1d", mutates_args=())
def _causal_conv1d(
    x: Tensor, weight: Tensor, bias: Tensor | None, activation: str | None, cu_seqlens: Tensor, chunk_indices: Tensor
) -> Tensor:
    return _conv_fwd(x, weight, bias, activation, cu_seqlens, chunk_indices, None)


@_causal_conv1d.register_fake
def _(x, weight, bias, activation, cu_seqlens, chunk_indices):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::fla_causal_conv1d_backward", mutates_args=())
def _causal_conv1d_backward(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation: str | None,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
) -> list[Tensor]:
    dx, dweight, dbias, _ = _conv_bwd(dy, x, weight, bias, activation, cu_seqlens, chunk_indices, None)
    return [dx, dweight] if bias is None else [dx, dweight, dbias.to(bias)]


@_causal_conv1d_backward.register_fake
def _(dy, x, weight, bias, activation, cu_seqlens, chunk_indices):
    grads = [torch.empty_like(x), torch.empty_like(weight)]
    return grads if bias is None else [*grads, torch.empty_like(bias)]


def _causal_conv1d_setup_context(ctx, inputs, output) -> None:
    x, weight, bias, activation, cu_seqlens, chunk_indices = inputs
    ctx.save_for_backward(x, weight, bias, cu_seqlens, chunk_indices)
    ctx.activation = activation


def _causal_conv1d_autograd(ctx, dy: Tensor):
    x, weight, bias, cu_seqlens, chunk_indices = ctx.saved_tensors
    dx, dweight, *dbias = _causal_conv1d_backward(
        dy.contiguous(), x, weight, bias, ctx.activation, cu_seqlens, chunk_indices
    )
    return dx, dweight, dbias[0] if dbias else None, None, None, None


_causal_conv1d.register_autograd(_causal_conv1d_autograd, setup_context=_causal_conv1d_setup_context)


@torch.library.custom_op("prime_rl::fla_causal_conv1d_cp", mutates_args=())
def _causal_conv1d_cp(
    x: Tensor,
    weight: Tensor,
    activation: str | None,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor]:
    cp_context = _fla_cp_context(cu_seqlens, cp_boundary, group_name)
    initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
        x=x, weight=weight, cu_seqlens=cu_seqlens, context=cp_context, group=cp_context.group
    )
    y = _conv_fwd(x, weight, None, activation, cu_seqlens, chunk_indices, initial_state)
    conv_state = x.new_zeros(weight.shape) if initial_state is None else initial_state[0].clone()
    return y, conv_state


@_causal_conv1d_cp.register_fake
def _(x, weight, activation, cu_seqlens, chunk_indices, cp_boundary, group_name):
    return torch.empty_like(x), x.new_empty(weight.shape)


@torch.library.custom_op("prime_rl::fla_causal_conv1d_cp_backward", mutates_args=())
def _causal_conv1d_cp_backward(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    conv_state: Tensor,
    activation: str | None,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor]:
    cp_context = _fla_cp_context(cu_seqlens, cp_boundary, group_name)
    initial_state = None
    if not cp_context.is_first_rank:
        initial_state = conv_state.new_zeros(cu_seqlens.shape[0] - 1, *conv_state.shape)
        initial_state[0] = conv_state
    dx, dweight, _, dh0 = _conv_bwd(dy, x, weight, None, activation, cu_seqlens, chunk_indices, initial_state)
    CausalConv1dFunctionCP._correct_dx_for_cp(
        dx=dx, dh0=dh0, W=weight.shape[-1], group=cp_context.group, cp_context=cp_context
    )
    return dx, dweight


@_causal_conv1d_cp_backward.register_fake
def _(dy, x, weight, conv_state, activation, cu_seqlens, chunk_indices, cp_boundary, group_name):
    return torch.empty_like(x), torch.empty_like(weight)


def _causal_conv1d_cp_setup_context(ctx, inputs, output) -> None:
    x, weight, activation, cu_seqlens, chunk_indices, cp_boundary, group_name = inputs
    _, conv_state = output
    ctx.mark_non_differentiable(conv_state)
    ctx.save_for_backward(x, weight, conv_state, cu_seqlens, chunk_indices, cp_boundary)
    ctx.activation = activation
    ctx.group_name = group_name


def _causal_conv1d_cp_autograd(ctx, dy: Tensor, _dconv_state: Tensor | None):
    x, weight, conv_state, cu_seqlens, chunk_indices, cp_boundary = ctx.saved_tensors
    dx, dweight = _causal_conv1d_cp_backward(
        dy.contiguous(), x, weight, conv_state, ctx.activation, cu_seqlens, chunk_indices, cp_boundary, ctx.group_name
    )
    return dx, dweight, None, None, None, None, None


_causal_conv1d_cp.register_autograd(_causal_conv1d_cp_autograd, setup_context=_causal_conv1d_cp_setup_context)


def causal_conv1d(
    x: Tensor, weight: Tensor, activation: str | None, varlen: FlaVarlen, cp_context: CPContext
) -> Tensor:
    """FLA's varlen causal depthwise convolution without bias or states: ``x`` is (1, tokens, channels), ``weight`` is
    (channels, width)."""
    x, weight = x.contiguous(), weight.contiguous()
    if not cp_context.cp_enabled:
        return _causal_conv1d(x, weight, None, activation, varlen.local_cu_seqlens, varlen.chunk_indices)
    return _causal_conv1d_cp(
        x,
        weight,
        activation,
        varlen.local_cu_seqlens,
        varlen.chunk_indices,
        varlen.cp_boundary,
        cp_context.cp_group.group_name,
    )[0]


def _gdn_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_context: FLACPContext | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    g_cumsum, o, A, _, h0, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    return o.to(q.dtype), q, q_rstd, k, k_rstd, g_cumsum, A, h0


def _gdn_bwd(
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
    h0: Tensor | None,
    cp_context: FLACPContext | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbeta, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=h0,
        do=do,
        dht=None,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    dq = l2norm_bwd(q, q_rstd, dq)
    dk = l2norm_bwd(k, k_rstd, dk)
    return dq.to(q), dk.to(k), dv.to(v), dg.to(g_cumsum), dbeta.to(beta)


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule", mutates_args=())
def _chunk_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, scale: float, cu_seqlens: Tensor, chunk_indices: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _gdn_fwd(q, k, v, g, beta, scale, cu_seqlens, chunk_indices, None)[:-1]


@_chunk_gated_delta_rule.register_fake
def _chunk_gated_delta_rule_fake(q, k, v, g, beta, scale, cu_seqlens, chunk_indices):
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
    return _gdn_bwd(do, q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, scale, cu_seqlens, chunk_indices, None, None)


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


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule_cp", mutates_args=())
def _chunk_gated_delta_rule_cp(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    cp_context = _fla_cp_context(cu_seqlens, cp_boundary, group_name)
    return _gdn_fwd(q, k, v, g, beta, scale, cu_seqlens, chunk_indices, cp_context)


@_chunk_gated_delta_rule_cp.register_fake
def _(q, k, v, g, beta, scale, cu_seqlens, chunk_indices, cp_boundary, group_name):
    return *_chunk_gated_delta_rule_fake(q, k, v, g, beta, scale, cu_seqlens, chunk_indices), _h0_fake(k, v)


@torch.library.custom_op("prime_rl::fla_chunk_gated_delta_rule_cp_backward", mutates_args=())
def _chunk_gated_delta_rule_cp_backward(
    do: Tensor,
    q: Tensor,
    q_rstd: Tensor,
    k: Tensor,
    k_rstd: Tensor,
    v: Tensor,
    g_cumsum: Tensor,
    beta: Tensor,
    A: Tensor,
    h0: Tensor,
    scale: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    cp_context = _fla_cp_context(cu_seqlens, cp_boundary, group_name)
    return _gdn_bwd(do, q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, scale, cu_seqlens, chunk_indices, h0, cp_context)


@_chunk_gated_delta_rule_cp_backward.register_fake
def _(do, q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, h0, scale, cu_seqlens, chunk_indices, cp_boundary, group_name):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g_cumsum),
        torch.empty_like(beta),
    )


def _chunk_gated_delta_rule_cp_setup_context(ctx, inputs, output) -> None:
    _, _, v, g, beta, scale, cu_seqlens, chunk_indices, cp_boundary, group_name = inputs
    _, q, q_rstd, k, k_rstd, g_cumsum, A, h0 = output
    ctx.mark_non_differentiable(q, q_rstd, k, k_rstd, g_cumsum, A, h0)
    ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, h0, cu_seqlens, chunk_indices, cp_boundary)
    ctx.scale = scale
    ctx.g_dtype = g.dtype
    ctx.group_name = group_name


def _chunk_gated_delta_rule_cp_autograd(ctx, do: Tensor, *_):
    q, q_rstd, k, k_rstd, v, g_cumsum, beta, A, h0, cu_seqlens, chunk_indices, cp_boundary = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = _chunk_gated_delta_rule_cp_backward(
        do.contiguous(),
        q,
        q_rstd,
        k,
        k_rstd,
        v,
        g_cumsum,
        beta,
        A,
        h0,
        ctx.scale,
        cu_seqlens,
        chunk_indices,
        cp_boundary,
        ctx.group_name,
    )
    return dq, dk, dv, dg.to(ctx.g_dtype), dbeta, None, None, None, None, None


_chunk_gated_delta_rule_cp.register_autograd(
    _chunk_gated_delta_rule_cp_autograd, setup_context=_chunk_gated_delta_rule_cp_setup_context
)


def chunk_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, varlen: FlaVarlen, cp_context: CPContext
) -> Tensor:
    """FLA's varlen ``chunk_gated_delta_rule`` with in-kernel q/k l2norm and no states: q, k and v are
    (1, tokens, heads, head_dim), g (log decay) and beta are (1, tokens, heads)."""
    args = (q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), k.shape[-1] ** -0.5)
    if not cp_context.cp_enabled:
        return _chunk_gated_delta_rule(*args, varlen.local_cu_seqlens, varlen.chunk_indices)[0]
    return _chunk_gated_delta_rule_cp(
        *args, varlen.local_cu_seqlens, varlen.chunk_indices, varlen.cp_boundary, cp_context.cp_group.group_name
    )[0]


def _kda_fwd(
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
    cp_context: FLACPContext | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    beta_sigmoid = fused_beta_sigmoid_fwd(beta)
    o, _, _, Aqk, Akk, *_, h0 = chunk_kda_fwd(
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
        cp_context=cp_context,
    )
    return o.type_as(q), q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk, h0


def _kda_bwd(
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
    h0: Tensor | None,
    cp_context: FLACPContext | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbeta_sigmoid, dg, _, dA_log, ddt_bias = chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        beta=beta_sigmoid,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=h0,
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
        cp_context=cp_context,
    )
    dq = l2norm_bwd(q, q_rstd, dq)
    dk = l2norm_bwd(k, k_rstd, dk)
    dbeta = fused_beta_sigmoid_bwd(x=beta, dy=dbeta_sigmoid)
    return dq.to(q), dk.to(k), dv.to(v), dg.to(g), dbeta.to(beta), dA_log.to(A_log), ddt_bias.to(dt_bias)


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
    return _kda_fwd(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices, None)[:-1]


@_chunk_kda.register_fake
def _chunk_kda_fake(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices):
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
    return _kda_bwd(
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
        None,
        None,
    )


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


@torch.library.custom_op("prime_rl::fla_chunk_kda_cp", mutates_args=())
def _chunk_kda_cp(
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
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    cp_context = _fla_cp_context(cu_seqlens, cp_boundary, group_name)
    return _kda_fwd(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices, cp_context)


@_chunk_kda_cp.register_fake
def _(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices, cp_boundary, group_name):
    return (
        *_chunk_kda_fake(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices),
        _h0_fake(k, v),
    )


@torch.library.custom_op("prime_rl::fla_chunk_kda_cp_backward", mutates_args=())
def _chunk_kda_cp_backward(
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
    h0: Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    cp_boundary: Tensor,
    group_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _kda_bwd(
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
        h0,
        _fla_cp_context(cu_seqlens, cp_boundary, group_name),
    )


@_chunk_kda_cp_backward.register_fake
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
    h0,
    scale,
    lower_bound,
    cu_seqlens,
    chunk_indices,
    cp_boundary,
    group_name,
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


def _chunk_kda_cp_setup_context(ctx, inputs, output) -> None:
    _, _, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens, chunk_indices, cp_boundary, group_name = inputs
    _, q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk, h0 = output
    ctx.mark_non_differentiable(q, q_rstd, k, k_rstd, beta_sigmoid, Aqk, Akk, h0)
    ctx.save_for_backward(
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
        h0,
        cu_seqlens,
        chunk_indices,
        cp_boundary,
    )
    ctx.scale = scale
    ctx.lower_bound = lower_bound
    ctx.group_name = group_name


def _chunk_kda_cp_autograd(ctx, do: Tensor, *_):
    (
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
        h0,
        cu_seqlens,
        chunk_indices,
        cp_boundary,
    ) = ctx.saved_tensors
    grads = _chunk_kda_cp_backward(
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
        h0,
        ctx.scale,
        ctx.lower_bound,
        cu_seqlens,
        chunk_indices,
        cp_boundary,
        ctx.group_name,
    )
    return *grads, None, None, None, None, None, None


_chunk_kda_cp.register_autograd(_chunk_kda_cp_autograd, setup_context=_chunk_kda_cp_setup_context)


def chunk_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    lower_bound: float,
    varlen: FlaVarlen,
    cp_context: CPContext,
) -> Tensor:
    """FLA's varlen ``chunk_kda`` with in-kernel q/k l2norm, safe gate and beta sigmoid, and no states: q, k, v and g
    (raw gate) are (1, tokens, heads, head_dim), beta (logits) is (1, tokens, heads)."""
    args = (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        k.shape[-1] ** -0.5,
        lower_bound,
    )
    if not cp_context.cp_enabled:
        return _chunk_kda(*args, varlen.local_cu_seqlens, varlen.chunk_indices)[0]
    return _chunk_kda_cp(
        *args, varlen.local_cu_seqlens, varlen.chunk_indices, varlen.cp_boundary, cp_context.cp_group.group_name
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
    return dx, dgate.to(gate), dweight


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
