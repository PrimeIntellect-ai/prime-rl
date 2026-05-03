"""Shared helpers for online MTP training.

The auxiliary CE path intentionally detaches trunk hidden states, token embeddings,
and lm-head weights so only MTP layers learn from this loss. The trunk and lm head
still update through the main SFT/RL objective.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MTPLossAutoScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output: Tensor, mtp_loss: Tensor) -> Tensor:  # type: ignore[override]
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return grad_output, None


def roll_tensor(
    tensor: Tensor, shift: int = -1, position_ids: Tensor | None = None, fill_value: int | bool = 0
) -> Tensor:
    if shift != -1:
        raise ValueError("MTP rolling currently supports shift=-1 only.")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a [batch, seq] tensor, got shape {tuple(tensor.shape)}.")

    rolled = torch.empty_like(tensor)
    rolled[:, :-1] = tensor[:, 1:]
    rolled[:, -1] = fill_value

    if position_ids is None:
        return rolled

    if position_ids.shape != tensor.shape:
        raise ValueError(f"position_ids shape {tuple(position_ids.shape)} does not match {tuple(tensor.shape)}.")

    next_is_same_sequence = position_ids[:, 1:] == position_ids[:, :-1] + 1
    rolled[:, :-1] = torch.where(next_is_same_sequence, rolled[:, :-1], torch.full_like(rolled[:, :-1], fill_value))
    return rolled


def make_viewless_tensor_with_grad(tensor: Tensor) -> Tensor:
    return tensor.detach().clone().requires_grad_(True)


def detached_lm_head_cross_entropy(
    lm_head: nn.Module,
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    if not hasattr(lm_head, "weight"):
        raise ValueError("MTP loss requires an lm_head with a weight parameter.")
    if hidden_states.shape[:2] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError(
            "MTP shape mismatch: "
            f"hidden={tuple(hidden_states.shape)}, labels={tuple(labels.shape)}, mask={tuple(loss_mask.shape)}."
        )

    logits = F.linear(hidden_states, lm_head.weight.detach())
    vocab_size = logits.shape[-1]
    labels = labels.masked_fill(~loss_mask, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    token_count = loss_mask.sum().clamp_min(1)
    return (token_loss * loss_mask).sum() / token_count


def mtp_masks_from_label_mask(loss_mask: Tensor, position_ids: Tensor | None, num_depths: int) -> Iterable[Tensor]:
    if loss_mask.dtype != torch.bool:
        loss_mask = loss_mask.bool()

    cumulative_mask = loss_mask
    shifted_mask = loss_mask
    for _ in range(num_depths):
        shifted_mask = roll_tensor(shifted_mask, position_ids=position_ids, fill_value=False)
        cumulative_mask = cumulative_mask & shifted_mask
        yield cumulative_mask
