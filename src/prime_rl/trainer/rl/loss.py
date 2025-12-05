from typing import Any

import torch
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
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits


def compute_loss(
    trainer_logprobs: list[Tensor],
    inference_logprobs: list[Tensor],
    advantages: list[Tensor],
    loss_mask: list[Tensor],
    loss_config: LossConfig,
) -> tuple[Tensor, dict[str, Any]]:
    """Compute loss for packed sequences."""
    cfg = loss_config
    total_loss = 0
    metrics = {
        "mismatch_kl": [],
        "masked_mismatch_kl": [],
        "unmasked_mismatch_kl": [],
        "token_masked": [],
        "token_masked_low": [],
        "token_masked_high": [],
        "seq_masked_low": [],
        "seq_masked_high": [],
        "seq_masked_neg_adv": [],
        "seq_masked_pos_adv": [],
    }

    for trainer_logprobs, inference_logprobs, advantages, loss_mask in zip(trainer_logprobs, inference_logprobs, advantages, loss_mask):
        log_ratio = trainer_logprobs - inference_logprobs
        token_kl = torch.exp(log_ratio) - log_ratio - 1

        seq_len = torch.clamp_min(loss_mask.sum(), 1)
        seq_log_ratio = log_ratio[loss_mask].sum()
        seq_ratio = torch.exp(seq_log_ratio / seq_len)
        seq_adv = advantages[loss_mask].mean()

        seq_mask_neg_adv = (seq_ratio < cfg.seq_mask_neg_adv) & (seq_adv < 0)
        seq_mask_pos_adv = (seq_ratio > cfg.seq_mask_pos_adv) & (seq_adv > 0)

        if cfg.ratio_type == "sequence":
            log_ratio = trainer_logprobs - trainer_logprobs.detach() + torch.clamp(seq_log_ratio, max=cfg.seq_clip).detach()
            seq_mask_low = seq_ratio < cfg.seq_mask_low
            seq_mask_high = seq_ratio > cfg.seq_mask_high
        else:
            ratio = torch.exp(log_ratio)
            min_ratio = ratio.masked_fill(~loss_mask, torch.inf).min()
            max_ratio = ratio.masked_fill(~loss_mask, -torch.inf).max()
            seq_mask_low = min_ratio < cfg.seq_mask_low
            seq_mask_high = max_ratio > cfg.seq_mask_high

        seq_mask = seq_mask_low | seq_mask_high | seq_mask_neg_adv | seq_mask_pos_adv

        ratio = torch.exp(log_ratio)
        token_mask_low = ratio < cfg.mask_low
        token_mask_high = ratio > cfg.mask_high
        token_mask = token_mask_low | token_mask_high | seq_mask

        keep = loss_mask & ~token_mask
        loss = (-ratio * advantages)[keep].sum() + cfg.kl_tau * log_ratio[loss_mask].sum()
        if cfg.ratio_type == "sequence":
            loss = loss / seq_len
        total_loss = total_loss + loss

        metrics["mismatch_kl"].append(token_kl[loss_mask].sum() / seq_len)
        metrics["masked_mismatch_kl"].append(token_kl[loss_mask & token_mask].sum() / torch.clamp_min((loss_mask & token_mask).sum(), 1))
        metrics["unmasked_mismatch_kl"].append(token_kl[keep].sum() / torch.clamp_min(keep.sum(), 1))
        metrics["token_masked"].append(token_mask[loss_mask].float())
        metrics["token_masked_low"].append(token_mask_low[loss_mask].float())
        metrics["token_masked_high"].append(token_mask_high[loss_mask].float())
        metrics["seq_masked_low"].append(seq_mask_low.float())
        metrics["seq_masked_high"].append(seq_mask_high.float())
        metrics["seq_masked_neg_adv"].append(seq_mask_neg_adv.float())
        metrics["seq_masked_pos_adv"].append(seq_mask_pos_adv.float())

    return total_loss, {
        "mismatch_kl": torch.stack(metrics["mismatch_kl"]),
        "masked_mismatch_kl": torch.stack(metrics["masked_mismatch_kl"]),
        "unmasked_mismatch_kl": torch.stack(metrics["unmasked_mismatch_kl"]),
        "token_masked": torch.cat(metrics["token_masked"]),
        "token_masked_low": torch.cat(metrics["token_masked_low"]),
        "token_masked_high": torch.cat(metrics["token_masked_high"]),
        "seq_masked_low": torch.stack(metrics["seq_masked_low"]),
        "seq_masked_high": torch.stack(metrics["seq_masked_high"]),
        "seq_masked_neg_adv": torch.stack(metrics["seq_masked_neg_adv"]),
        "seq_masked_pos_adv": torch.stack(metrics["seq_masked_pos_adv"]),
    }