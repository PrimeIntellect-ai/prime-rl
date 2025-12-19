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
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def compute_loss(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    reference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
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
        reference_logprobs: Reference log probabilities tensor for packed sequences
        advantages: Advantages tensor for packed sequences
        loss_mask: Loss mask tensor for packed sequences
        loss_config: Loss configuration object
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_loss_tensors)
    """

    total_loss = 0.0
    total_mismatch_kl = []
    total_masked_mismatch_kl = []
    total_unmasked_mismatch_kl = []
    total_is_masked = []
    total_is_masked_low = []
    total_is_masked_high = []
    total_sequence_masked_low = []
    total_sequence_masked_high = []
    total_geo_masked_low = []
    total_geo_masked_high = []
    total_geo_seq_ratio = []

    for trainer_logprobs, inference_logprobs, reference_logprobs, advantages, loss_mask in zip(
        trainer_logprobs, inference_logprobs, reference_logprobs, advantages, loss_mask
    ):
        log_importance_ratio = trainer_logprobs - inference_logprobs
        ref_kl = trainer_logprobs - reference_logprobs

        # Trainer-inference mismatch KL per token
        token_importance_ratio = torch.exp(log_importance_ratio)
        geo_seq_ratio = torch.exp(_safe_mean(log_importance_ratio, loss_mask))
        token_mismatch_kl = token_importance_ratio - log_importance_ratio - 1

        seq_log_importance_ratio = torch.clamp(log_importance_ratio[loss_mask].sum().detach(), max=10.0)
        seq_importance_ratio = torch.clamp(torch.exp(seq_log_importance_ratio), max=loss_config.sequence_clip_high)

        seq_min_ratio = torch.where(loss_mask, token_importance_ratio, torch.inf).min()
        seq_max_ratio = torch.where(loss_mask, token_importance_ratio, -torch.inf).max()
        seq_mask_low = seq_min_ratio < loss_config.sequence_mask_low
        seq_mask_high = seq_max_ratio > loss_config.sequence_mask_high

        token_mask_low = token_importance_ratio < loss_config.token_mask_low
        token_mask_high = token_importance_ratio > loss_config.token_mask_high

        geo_mask_low = geo_seq_ratio < loss_config.geo_mask_low
        geo_mask_high = geo_seq_ratio > loss_config.geo_mask_high

        is_masked = token_mask_low | token_mask_high | geo_mask_low | geo_mask_high | seq_mask_low | seq_mask_high
        keep_mask = loss_mask & ~is_masked

        importance_ratio = seq_importance_ratio if loss_config.ratio_type == "sequence" else token_importance_ratio

        advantages = loss_config.adv_tau * advantages + loss_config.ref_tau * ref_kl.detach()
        coeff = importance_ratio * (advantages - loss_config.kl_tau * log_importance_ratio)
        loss = -(coeff.detach() * trainer_logprobs)[keep_mask].sum()

        if loss_config.ratio_type == "sequence":
            loss = loss / torch.clamp_min(loss_mask.sum(), 1)

        total_loss = total_loss + loss

        # Aggregate loss tensors
        total_mismatch_kl.append(_safe_mean(token_mismatch_kl, loss_mask))
        total_masked_mismatch_kl.append(_safe_mean(token_mismatch_kl, loss_mask & is_masked))
        total_unmasked_mismatch_kl.append(_safe_mean(token_mismatch_kl, keep_mask))
        total_is_masked.append(is_masked[loss_mask].float())
        total_is_masked_low.append(token_mask_low[loss_mask].float())
        total_is_masked_high.append(token_mask_high[loss_mask].float())
        total_sequence_masked_low.append(seq_mask_low.float())
        total_sequence_masked_high.append(seq_mask_high.float())
        total_geo_masked_low.append(geo_mask_low.float())
        total_geo_masked_high.append(geo_mask_high.float())
        total_geo_seq_ratio.append(geo_seq_ratio)

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
        "sequence_masked_high": torch.stack(total_sequence_masked_high),
        "geo_masked_low": torch.stack(total_geo_masked_low),
        "geo_masked_high": torch.stack(total_geo_masked_high),
        "geo_seq_ratio": torch.stack(total_geo_seq_ratio),
    }
