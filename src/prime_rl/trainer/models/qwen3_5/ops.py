import torch
from fla.modules.conv import causal_conv1d as fla_causal_conv1d
from fla.modules.conv.triton.ops import _has_non_standard_layout, causal_conv1d_bwd
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.utils.index import prepare_chunk_indices


@torch.library.custom_op("prime_rl_qwen3_5::causal_conv1d", mutates_args=())
def _causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    output, _ = fla_causal_conv1d(
        x=x,
        weight=weight,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    return output


@_causal_conv1d.register_fake
def _causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("prime_rl_qwen3_5::causal_conv1d_backward", mutates_args=())
def _causal_conv1d_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    grad_output: torch.Tensor,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_weight, _, _, _ = causal_conv1d_bwd(
        x=x,
        dy=grad_output,
        dht=None,
        weight=weight,
        activation=activation,
        cu_seqlens=cu_seqlens,
        layout_fallback=_has_non_standard_layout(x),
    )
    return grad_x, grad_weight


@_causal_conv1d_backward.register_fake
def _causal_conv1d_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    grad_output: torch.Tensor,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _causal_conv1d_setup_context(ctx, inputs, output) -> None:
    x, weight, cu_seqlens, activation = inputs
    ctx.save_for_backward(x, weight, cu_seqlens)
    ctx.activation = activation


def _causal_conv1d_autograd_backward(ctx, grad_output: torch.Tensor):
    x, weight, cu_seqlens = ctx.saved_tensors
    grad_x, grad_weight = _causal_conv1d_backward(x, weight, cu_seqlens, grad_output, ctx.activation)
    return grad_x, grad_weight, None, None


_causal_conv1d.register_autograd(
    _causal_conv1d_autograd_backward,
    setup_context=_causal_conv1d_setup_context,
)


@torch.library.custom_op("prime_rl_qwen3_5::chunk_gated_delta_rule", mutates_args=())
def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # The low-level FLA kernels require the layout enforced by their public wrapper's input_guard.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    decay = decay.contiguous()
    beta = beta.contiguous()
    normalized_query, query_rstd = l2norm_fwd(query)
    normalized_key, key_rstd = l2norm_fwd(key)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    cumulative_decay, output, matrix, _, _, _ = chunk_gated_delta_rule_fwd(
        q=normalized_query,
        k=normalized_key,
        v=value,
        g=decay,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
    )
    return (
        output.to(query.dtype),
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        cumulative_decay,
        matrix,
    )


@_chunk_gated_delta_rule.register_fake
def _chunk_gated_delta_rule_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    matrix_shape = (*key.shape[:2], value.shape[2], 64)
    return (
        torch.empty_like(value),
        torch.empty_like(query),
        query.new_empty(query.shape[:-1], dtype=torch.float32),
        torch.empty_like(key),
        key.new_empty(key.shape[:-1], dtype=torch.float32),
        torch.empty_like(decay),
        key.new_empty(matrix_shape),
    )


@torch.library.custom_op("prime_rl_qwen3_5::chunk_gated_delta_rule_backward", mutates_args=())
def _chunk_gated_delta_rule_backward(
    normalized_query: torch.Tensor,
    query_rstd: torch.Tensor,
    normalized_key: torch.Tensor,
    key_rstd: torch.Tensor,
    value: torch.Tensor,
    cumulative_decay: torch.Tensor,
    beta: torch.Tensor,
    matrix: torch.Tensor,
    cu_seqlens: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    grad_query, grad_key, grad_value, grad_beta, grad_decay, _, _, _ = chunk_gated_delta_rule_bwd(
        q=normalized_query,
        k=normalized_key,
        v=value,
        g=cumulative_decay,
        beta=beta,
        A=matrix,
        scale=scale,
        initial_state=None,
        do=grad_output,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
    )
    grad_query = l2norm_bwd(normalized_query, query_rstd, grad_query)
    grad_key = l2norm_bwd(normalized_key, key_rstd, grad_key)
    return grad_query, grad_key, grad_value, grad_decay, grad_beta


@_chunk_gated_delta_rule_backward.register_fake
def _chunk_gated_delta_rule_backward_fake(
    normalized_query: torch.Tensor,
    query_rstd: torch.Tensor,
    normalized_key: torch.Tensor,
    key_rstd: torch.Tensor,
    value: torch.Tensor,
    cumulative_decay: torch.Tensor,
    beta: torch.Tensor,
    matrix: torch.Tensor,
    cu_seqlens: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(normalized_query),
        torch.empty_like(normalized_key),
        torch.empty_like(value),
        torch.empty_like(cumulative_decay),
        torch.empty_like(beta),
    )


def _chunk_gated_delta_rule_setup_context(ctx, inputs, output) -> None:
    _, _, value, _, beta, cu_seqlens, scale = inputs
    (
        _,
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        cumulative_decay,
        matrix,
    ) = output
    ctx.save_for_backward(
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        value,
        cumulative_decay,
        beta,
        matrix,
        cu_seqlens,
    )
    ctx.scale = scale
    ctx.mark_non_differentiable(
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        cumulative_decay,
        matrix,
    )


def _chunk_gated_delta_rule_autograd_backward(
    ctx,
    grad_output: torch.Tensor,
    _grad_normalized_query: torch.Tensor | None,
    _grad_query_rstd: torch.Tensor | None,
    _grad_normalized_key: torch.Tensor | None,
    _grad_key_rstd: torch.Tensor | None,
    _grad_cumulative_decay: torch.Tensor | None,
    _grad_matrix: torch.Tensor | None,
):
    (
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        value,
        cumulative_decay,
        beta,
        matrix,
        cu_seqlens,
    ) = ctx.saved_tensors
    gradients = _chunk_gated_delta_rule_backward(
        normalized_query,
        query_rstd,
        normalized_key,
        key_rstd,
        value,
        cumulative_decay,
        beta,
        matrix,
        cu_seqlens,
        grad_output,
        ctx.scale,
    )
    return *gradients, None, None


_chunk_gated_delta_rule.register_autograd(
    _chunk_gated_delta_rule_autograd_backward,
    setup_context=_chunk_gated_delta_rule_setup_context,
)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return _causal_conv1d(x, weight, cu_seqlens, activation)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    output, *_ = _chunk_gated_delta_rule(query, key, value, decay, beta, cu_seqlens, scale)
    return output


__all__ = ["causal_conv1d", "chunk_gated_delta_rule"]
