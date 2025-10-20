from typing import Any

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from prime_rl.trainer.rl.config import LossConfig


@jaxtyped(typechecker=typechecker)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    """
    credits to https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/utils.py#L1659

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@jaxtyped(typechecker=typechecker)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
    entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits


def compute_loss(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    advantages: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_mask: Any,  # list of Bool[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_config: LossConfig,
    loss_scale: int,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities tensor for packed sequences
        inference_logprobs: Old log probabilities tensor for packed sequences
        advantages: Advantages tensor for packed sequences
        loss_mask: Loss mask tensor for packed sequences
        loss_config: Loss configuration object
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_loss_tensors)
    """

    total_loss = 0
    total_mismatch_kl = []
    total_masked_mismatch_kl = []
    total_unmasked_mismatch_kl = []
    total_is_masked = []
    total_is_masked_low = []
    total_is_masked_high = []

    for trainer_logprobs, inference_logprobs, advantages, loss_mask in zip(
        trainer_logprobs, inference_logprobs, advantages, loss_mask
    ):
        log_importance_ratio = trainer_logprobs - inference_logprobs

        # Compute trainer-inference mismatch KL
        token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        if loss_config.ratio_type == "sequence":
            seq_log_importance_ratio = (log_importance_ratio[loss_mask]).sum()
            if loss_config.ratio_length_norm:
                seq_log_importance_ratio = seq_log_importance_ratio / torch.clamp_min(loss_mask.sum(), 1)
            log_importance_ratio = trainer_logprobs - trainer_logprobs.detach() + seq_log_importance_ratio.detach()
            log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)

        importance_ratio = torch.exp(log_importance_ratio)
        is_masked_low = importance_ratio < loss_config.mask_ratio_low
        is_masked_high = importance_ratio > loss_config.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        keep_mask = loss_mask & ~is_masked
        loss = (-importance_ratio * advantages)[keep_mask].sum()

        # Apply sequence-level normalization if configured
        if loss_config.ratio_type == "sequence":
            loss = loss / torch.clamp_min(loss_mask.sum(), 1)

        total_loss = total_loss + loss

        mismatch_kl = token_mismatch_kl[loss_mask].sum() / torch.clamp_min(loss_mask.sum(), 1)
        masked_mismatch_kl = token_mismatch_kl[loss_mask & is_masked].sum() / torch.clamp_min(
            (loss_mask & is_masked).sum(), 1
        )
        unmasked_mismatch_kl = token_mismatch_kl[keep_mask].sum() / torch.clamp_min(keep_mask.sum(), 1)

        # Aggregate loss tensors
        total_mismatch_kl.append(mismatch_kl)
        total_masked_mismatch_kl.append(masked_mismatch_kl)
        total_unmasked_mismatch_kl.append(unmasked_mismatch_kl)
        total_is_masked.append(is_masked[loss_mask].float())
        total_is_masked_low.append(is_masked_low[loss_mask].float())
        total_is_masked_high.append(is_masked_high[loss_mask].float())

    # Apply loss scaling
    scaled_loss = total_loss / loss_scale

    return scaled_loss, {
        "mismatch_kl": torch.stack(total_mismatch_kl),
        "masked_mismatch_kl": torch.stack(total_masked_mismatch_kl),
        "unmasked_mismatch_kl": torch.stack(total_unmasked_mismatch_kl),
        "is_masked": torch.cat(total_is_masked),
        "is_masked_low": torch.cat(total_is_masked_low),
        "is_masked_high": torch.cat(total_is_masked_high),
    }
