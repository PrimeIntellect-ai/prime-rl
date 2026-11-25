"""Shared categorical log-likelihood primitives for policy and supervised actors."""

import torch
from torch import Tensor


def gather_action_logprobs(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def summed_weighted_nll(logprobs: Tensor, loss_mask: Tensor | None, loss_weights: Tensor | None) -> Tensor:
    nll = -logprobs
    if loss_weights is not None:
        nll = nll * loss_weights
    if loss_mask is not None:
        nll = nll[loss_mask]
    return nll.sum()
