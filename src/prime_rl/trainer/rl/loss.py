from typing import Any

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is not None:
        # left pad logit is not None if this is not the first CP rank, in which case we use the last logit from the previous rank as the left pad
        zeros = left_pad_logit
    else:
        zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits


def compute_sequence_stats(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """
    Compute stats for a single sequence needed for CP sync.
    Returns: Tensor[3] -> [sum_log_importance_ratio, sum_loss_mask, min_log_importance_ratio]
    """
    log_importance_ratio = trainer_logprobs - inference_logprobs
    
    sum_log_imp = (log_importance_ratio[loss_mask]).sum()
    sum_mask = loss_mask.sum()
    min_log_imp = log_importance_ratio.masked_fill(~loss_mask, torch.inf).min()
    
    return torch.stack([sum_log_imp, sum_mask, min_log_imp])


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
    total_sequence_masked_low = []

    for trainer_logprobs, inference_logprobs, advantages, loss_mask in zip(
        trainer_logprobs, inference_logprobs, advantages, loss_mask
    ):
        log_importance_ratio = trainer_logprobs - inference_logprobs

        # Compute trainer-inference mismatch KL
        token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        if loss_config.ratio_type == "sequence":
            seq_log_importance_ratio = (log_importance_ratio[loss_mask]).sum()
            log_importance_ratio = trainer_logprobs - trainer_logprobs.detach() + seq_log_importance_ratio.detach()
            log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)

        importance_ratio = torch.exp(log_importance_ratio)
        is_masked_low = importance_ratio < loss_config.mask_ratio_low
        is_masked_high = importance_ratio > loss_config.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        seq_min_ratio = importance_ratio.masked_fill(~loss_mask, torch.inf).min()
        seq_should_mask = seq_min_ratio < loss_config.sequence_mask_ratio_low
        is_masked = is_masked | seq_should_mask
        keep_mask = loss_mask & ~is_masked
        loss = (-importance_ratio * advantages)[keep_mask].sum()
        if loss_config.kl_mask_type == "masked":
            kl_mask = loss_mask & is_masked
        elif loss_config.kl_mask_type == "unmasked":
            kl_mask = keep_mask
        elif loss_config.kl_mask_type == "all":
            kl_mask = loss_mask
        else:
            raise ValueError(f"Invalid KL mask type: {loss_config.kl_mask_type}")
        loss = loss + loss_config.kl_tau * (log_importance_ratio[kl_mask]).sum()

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
        total_sequence_masked_low.append(seq_should_mask.float())

    # Apply loss scaling
    scaled_loss = total_loss / loss_scale

    return scaled_loss, {
        "mismatch_kl": torch.stack(total_mismatch_kl),
        "masked_mismatch_kl": torch.stack(total_masked_mismatch_kl),
        "unmasked_mismatch_kl": torch.stack(total_unmasked_mismatch_kl),
        "is_masked": torch.cat(total_is_masked),
        "is_masked_low": torch.cat(total_is_masked_low),
        "is_masked_high": torch.cat(total_is_masked_high),
        "sequence_masked_low": torch.stack(total_sequence_masked_low),
    }
