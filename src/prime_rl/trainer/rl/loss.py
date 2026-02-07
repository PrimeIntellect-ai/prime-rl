from pathlib import Path
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


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def compute_loss(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    teacher_logprobs: Any | None,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths, or None
    advantages: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_mask: Any,  # list of Bool[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_config: LossConfig,
    loss_scale: int,
    step: int | None = None,
    rank: int | None = None,
    output_dir: Path | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities tensor for packed sequences
        inference_logprobs: Old log probabilities tensor for packed sequences
        teacher_logprobs: Teacher log probabilities tensor for packed sequences, or None if not configured
        advantages: Advantages tensor for packed sequences
        loss_mask: Loss mask tensor for packed sequences
        loss_config: Loss configuration object
        loss_scale: Scale factor to normalize the loss
        step: Optional step number for saving debug tensors
        rank: Optional rank number for saving debug tensors

    Returns:
        Tuple of (scaled_loss, aggregated_loss_tensors)
    """
    # Save debug tensors if step and rank are provided
    if step is not None and rank is not None and step % 10 == 0:
        assert output_dir is not None
        output_dir = output_dir / "loss" / f"step_{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"rank_{rank}.bin"
        torch.save(
            {
                "trainer_logprobs": [t.detach().cpu() for t in trainer_logprobs],
                "inference_logprobs": [t.detach().cpu() for t in inference_logprobs],
                "advantages": [t.detach().cpu() for t in advantages],
                "loss_mask": [t.detach().cpu() for t in loss_mask],
            },
            output_file,
        )

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
    total_teacher_kl = []
    total_is_clipped_low = []
    total_is_clipped_high = []
    total_kl_loss = []

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    for trainer_logprobs, inference_logprobs, teacher_logprobs, advantages, loss_mask in zip(
        trainer_logprobs, inference_logprobs, teacher_logprobs, advantages, loss_mask
    ):
        log_importance_ratio = trainer_logprobs - inference_logprobs
        teacher_kl = teacher_logprobs - trainer_logprobs if teacher_logprobs is not None else None

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

        # DPPO-Bin-TV: mask conditionally on advantage (PPO-style)
        prob = torch.exp(trainer_logprobs)
        rollout_prob = torch.exp(inference_logprobs)
        invalid_tv_positive = (prob - rollout_prob) > loss_config.dppo_tv_clip_high
        invalid_tv_negative = (prob - rollout_prob) < -loss_config.dppo_tv_clip_low
        invalid_dppo_tv = torch.where(advantages > 0, invalid_tv_positive, invalid_tv_negative)

        # PPO: mask when ratio outside [1 - clip_low, 1 + clip_high], conditionally on advantage
        invalid_ppo_positive = token_importance_ratio > (1.0 + loss_config.ppo_clip_high)
        invalid_ppo_negative = token_importance_ratio < (1.0 - loss_config.ppo_clip_low)
        invalid_ppo = torch.where(advantages > 0, invalid_ppo_positive, invalid_ppo_negative)

        is_masked = (
            token_mask_low
            | token_mask_high
            | invalid_dppo_tv
            | invalid_ppo
            | geo_mask_low
            | geo_mask_high
            | seq_mask_low
            | seq_mask_high
        )
        keep_mask = loss_mask & ~is_masked

        # tis: clamp is ratio (independent of masking)
        token_clip_low = token_importance_ratio < loss_config.token_clip_low
        token_clip_high = token_importance_ratio > loss_config.token_clip_high
        clipped_token_importance_ratio = torch.clamp(
            token_importance_ratio, min=loss_config.token_clip_low, max=loss_config.token_clip_high
        )

        importance_ratio = (
            seq_importance_ratio if loss_config.ratio_type == "sequence" else clipped_token_importance_ratio
        )
        if not loss_config.use_importance_sampling_ratio:
            importance_ratio = torch.ones_like(importance_ratio)

        advantages = loss_config.adv_tau * advantages
        if teacher_logprobs is not None:
            advantages = advantages + loss_config.teacher_tau * teacher_kl.detach()
        coeff = importance_ratio * advantages * keep_mask
        pg_loss = coeff.detach() * trainer_logprobs
        kl_loss = log_importance_ratio**2
        loss = (loss_config.kl_tau * kl_loss * loss_mask - pg_loss).sum()

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
        total_is_clipped_low.append(_safe_mean(token_clip_low, keep_mask))
        total_is_clipped_high.append(_safe_mean(token_clip_high, keep_mask))
        total_kl_loss.append(kl_loss[loss_mask])
        if teacher_logprobs is not None:
            total_teacher_kl.append(_safe_mean(teacher_kl, loss_mask))

    # Apply loss scaling
    scaled_loss = total_loss / loss_scale

    result = {
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
        "is_clipped_low": torch.stack(total_is_clipped_low),
        "is_clipped_high": torch.stack(total_is_clipped_high),
        "kl_loss": torch.cat(total_kl_loss),
    }
    if total_teacher_kl:
        result["teacher_kl"] = torch.stack(total_teacher_kl)
    return scaled_loss, result
