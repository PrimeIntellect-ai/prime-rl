import torch
from fla.modules.conv.triton.ops import _has_non_standard_layout, causal_conv1d_bwd, causal_conv1d_fwd
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd


@torch.library.custom_op("prime_rl_qwen3_5::causal_conv1d", mutates_args=())
def _causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    output, _ = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout_fallback=_has_non_standard_layout(x),
    )
    return output


@_causal_conv1d.register_fake
def _causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("prime_rl_qwen3_5::causal_conv1d_backward", mutates_args=())
def _causal_conv1d_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_weight, _, _, _ = causal_conv1d_bwd(
        x=x,
        dy=grad_output,
        dht=None,
        weight=weight,
        activation=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout_fallback=_has_non_standard_layout(x),
    )
    return grad_x, grad_weight


@_causal_conv1d_backward.register_fake
def _causal_conv1d_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _causal_conv1d_setup_context(ctx, inputs, output) -> None:
    x, weight, cu_seqlens, chunk_indices = inputs
    ctx.save_for_backward(x, weight, cu_seqlens, chunk_indices)


def _causal_conv1d_autograd_backward(ctx, grad_output: torch.Tensor):
    x, weight, cu_seqlens, chunk_indices = ctx.saved_tensors
    grad_x, grad_weight = _causal_conv1d_backward(x, weight, cu_seqlens, chunk_indices, grad_output)
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
    chunk_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # The low-level FLA kernels require the layout enforced by their public wrapper's input_guard.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    decay = decay.contiguous()
    beta = beta.contiguous()
    normalized_query, query_rstd = l2norm_fwd(query)
    normalized_key, key_rstd = l2norm_fwd(key)
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
    return output.to(query.dtype)


@_chunk_gated_delta_rule.register_fake
def _chunk_gated_delta_rule_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(value)


@torch.library.custom_op("prime_rl_qwen3_5::chunk_gated_delta_rule_backward", mutates_args=())
def _chunk_gated_delta_rule_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    decay = decay.contiguous()
    beta = beta.contiguous()
    grad_output = grad_output.contiguous()
    normalized_query, query_rstd = l2norm_fwd(query)
    normalized_key, key_rstd = l2norm_fwd(key)
    cumulative_decay, _, matrix, _, _, _ = chunk_gated_delta_rule_fwd(
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
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
        torch.empty_like(decay),
        torch.empty_like(beta),
    )


def _chunk_gated_delta_rule_setup_context(ctx, inputs, output) -> None:
    query, key, value, decay, beta, cu_seqlens, chunk_indices, scale = inputs
    ctx.save_for_backward(query, key, value, decay, beta, cu_seqlens, chunk_indices)
    ctx.scale = scale


def _chunk_gated_delta_rule_autograd_backward(
    ctx,
    grad_output: torch.Tensor,
):
    query, key, value, decay, beta, cu_seqlens, chunk_indices = ctx.saved_tensors
    gradients = _chunk_gated_delta_rule_backward(
        query,
        key,
        value,
        decay,
        beta,
        cu_seqlens,
        chunk_indices,
        grad_output,
        ctx.scale,
    )
    return *gradients, None, None, None


_chunk_gated_delta_rule.register_autograd(
    _chunk_gated_delta_rule_autograd_backward,
    setup_context=_chunk_gated_delta_rule_setup_context,
)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    output = _causal_conv1d(x, weight, cu_seqlens, chunk_indices)
    if activation in ("silu", "swish"):
        return torch.nn.functional.silu(output)
    if activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")
    return output


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    return _chunk_gated_delta_rule(
        query,
        key,
        value,
        decay,
        beta,
        cu_seqlens,
        chunk_indices,
        scale,
    )


__all__ = ["causal_conv1d", "chunk_gated_delta_rule"]
