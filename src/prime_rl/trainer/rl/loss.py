from typing import Any

import torch
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


@torch.compile(dynamic=True)
def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Tensor) -> Tensor:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


def shift_logits(logits: Tensor) -> Tensor:
    """Removes final token logits and adds a zero logit for the first token."""
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)
    return torch.cat([zeros, logits], dim=1)


def compute_loss(
    trainer_logprobs: list[Tensor],
    inference_logprobs: list[Tensor],
    advantages: list[Tensor],
    loss_mask: list[Tensor],
    loss_config: LossConfig,
    loss_scale: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Compute loss for packed sequences."""
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
        log_ratio = trainer_logprobs - inference_logprobs
        token_mismatch_kl = torch.exp(log_ratio) - log_ratio - 1

        if loss_config.ratio_type == "sequence":
            seq_log_ratio = log_ratio[loss_mask].sum()
            log_ratio = trainer_logprobs - trainer_logprobs.detach() + seq_log_ratio.detach()
            log_ratio = torch.clamp(log_ratio, max=10.0)

        ratio = torch.exp(log_ratio)
        is_masked_low = ratio < loss_config.mask_ratio_low
        is_masked_high = ratio > loss_config.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        seq_min_ratio = ratio.masked_fill(~loss_mask, torch.inf).min()
        seq_should_mask = seq_min_ratio < loss_config.sequence_mask_ratio_low
        is_masked = is_masked | seq_should_mask
        keep_mask = loss_mask & ~is_masked

        loss = (-ratio * advantages)[keep_mask].sum() + loss_config.kl_tau * log_ratio[loss_mask].sum()

        if loss_config.ratio_type == "sequence":
            loss = loss / torch.clamp_min(loss_mask.sum(), 1)

        total_loss = total_loss + loss

        mismatch_kl = token_mismatch_kl[loss_mask].sum() / torch.clamp_min(loss_mask.sum(), 1)
        masked_mismatch_kl = token_mismatch_kl[loss_mask & is_masked].sum() / torch.clamp_min((loss_mask & is_masked).sum(), 1)
        unmasked_mismatch_kl = token_mismatch_kl[keep_mask].sum() / torch.clamp_min(keep_mask.sum(), 1)

        total_mismatch_kl.append(mismatch_kl)
        total_masked_mismatch_kl.append(masked_mismatch_kl)
        total_unmasked_mismatch_kl.append(unmasked_mismatch_kl)
        total_is_masked.append(is_masked[loss_mask].float())
        total_is_masked_low.append(is_masked_low[loss_mask].float())
        total_is_masked_high.append(is_masked_high[loss_mask].float())
        total_sequence_masked_low.append(seq_should_mask[loss_mask].float())

    return total_loss / loss_scale, {
        "mismatch_kl": torch.stack(total_mismatch_kl),
        "masked_mismatch_kl": torch.stack(total_masked_mismatch_kl),
        "unmasked_mismatch_kl": torch.stack(total_unmasked_mismatch_kl),
        "is_masked": torch.cat(total_is_masked),
        "is_masked_low": torch.cat(total_is_masked_low),
        "is_masked_high": torch.cat(total_is_masked_high),
        "sequence_masked_low": torch.stack(total_sequence_masked_low),
    }
