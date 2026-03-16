"""
Vendored and modified version of Liger's fused_linear_cross_entropy_forward that supports
per-token weighting for sample-level loss normalization.

The Triton kernel (liger_cross_entropy_kernel) is NOT modified — only the Python wrapper
that calls it and accumulates gradients. The key changes:
1. Accepts an optional `token_weight` tensor [BT] that scales each token's loss and gradient
2. Uses reduction="sum" internally, then applies token_weight to get the sample-weighted loss
"""

import torch
import triton
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, is_hip

MAX_FUSED_SIZE = 65536 // 2


def _fused_linear_cross_entropy_forward_with_token_weight(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor,
    ignore_index: int = -100,
    softcap: float | None = None,
):
    """Fused linear + cross-entropy forward with per-token weighting.

    Like Liger's fused_linear_cross_entropy_forward but:
    - Always uses reduction="sum" internally
    - Applies token_weight to both loss and pre-computed gradients
    - Returns weighted scalar loss

    Args:
        _input: [BT, H] hidden states
        weight: [V, H] lm_head weight
        target: [BT] target token ids (ignore_index for masked positions)
        token_weight: [BT] per-token weight (e.g. 1/|O_c| for sample normalization)
        ignore_index: target value to ignore
        softcap: optional logit soft-capping value
    """
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_input = torch.zeros_like(_input, device=device)
    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()
        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Triton kernel computes per-token loss and overwrites logits_chunk with gradients.
        # Using reduction="sum" so gradients are (softmax - one_hot), not divided by N.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=None,
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_n_non_ignore,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction="sum",
            softcap=softcap,
            RETURN_Z_LOSS=False,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=softcap is not None,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        grad_logits_chunk = logits_chunk  # kernel overwrote logits in-place with grads

        # Apply per-token weight to gradients before projecting back to input space.
        # This is mathematically equivalent to scaling after projection (linearity).
        weight_chunk = token_weight[start_idx:end_idx].unsqueeze(-1)
        grad_logits_chunk = grad_logits_chunk * weight_chunk

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

    loss = torch.sum(loss_1d * token_weight)

    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    return loss, grad_input, grad_weight


class SampleWeightedFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(ctx, _input, weight, target, token_weight, ignore_index, softcap):
        loss, grad_input, grad_weight = _fused_linear_cross_entropy_forward_with_token_weight(
            _input, weight, target, token_weight, ignore_index, softcap
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight, _ = fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, None)
        return grad_input, grad_weight, None, None, None, None


def sample_weighted_fused_cross_entropy(
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    token_weight: torch.Tensor,
    ignore_index: int = -100,
    softcap: float | None = None,
) -> torch.Tensor:
    """Compute fused linear + cross-entropy loss with per-token sample weights.

    Args:
        weight: [V, H] lm_head weight matrix
        hidden_states: [B, S, H] hidden states from model backbone
        labels: [B, S] target ids (ignore_index for positions to skip)
        token_weight: [B, S] per-token weight for sample normalization
        ignore_index: target value to ignore (default -100)
        softcap: optional logit soft-capping value

    Returns:
        Scalar loss: sum of (per-token loss * token_weight)
    """
    b, s, h = hidden_states.shape
    hidden_flat = hidden_states.reshape(b * s, h).contiguous()
    labels_flat = labels.reshape(b * s).contiguous()
    token_weight_flat = token_weight.reshape(b * s).contiguous()
    return SampleWeightedFusedLinearCrossEntropyFunction.apply(
        hidden_flat, weight, labels_flat, token_weight_flat, ignore_index, softcap
    )
