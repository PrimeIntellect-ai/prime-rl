from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.trainer.rl.config import CustomLossConfig, LossConfig, LossConfigType
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


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


def default_loss_fn(inputs: LossInputs, loss_config: LossConfig) -> LossOutputs:
    """Masked importance sampling with KL against the inference policy, and optional masking strategies."""
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio = trainer_logprobs - inference_logprobs
    teacher_kl = teacher_logprobs - trainer_logprobs if teacher_logprobs is not None else None

    token_importance_ratio = torch.exp(log_importance_ratio)
    geo_seq_ratio = torch.exp(_safe_mean(log_importance_ratio, loss_mask))
    token_mismatch_kl = token_importance_ratio - log_importance_ratio - 1

    seq_log_importance_ratio = torch.clamp(log_importance_ratio[loss_mask].sum().detach(), max=10.0)
    seq_importance_ratio = torch.clamp(torch.exp(seq_log_importance_ratio), max=loss_config.sequence_clip_high)

    seq_min_ratio = torch.where(loss_mask, token_importance_ratio, torch.inf).min()
    seq_max_ratio = torch.where(loss_mask, token_importance_ratio, -torch.inf).max()
    seq_mask_low = seq_min_ratio < loss_config.sequence_mask_low
    seq_mask_high = seq_max_ratio > loss_config.sequence_mask_high

    token_mask_low_mask = token_importance_ratio < loss_config.token_mask_low
    token_mask_high_mask = token_importance_ratio > loss_config.token_mask_high

    geo_mask_low_mask = geo_seq_ratio < loss_config.geo_mask_low
    geo_mask_high_mask = geo_seq_ratio > loss_config.geo_mask_high

    is_masked = (
        token_mask_low_mask
        | token_mask_high_mask
        | geo_mask_low_mask
        | geo_mask_high_mask
        | seq_mask_low
        | seq_mask_high
    )
    keep_mask = loss_mask & ~is_masked

    importance_ratio = seq_importance_ratio if loss_config.ratio_type == "sequence" else token_importance_ratio

    advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        advantages = advantages + loss_config.teacher_tau * teacher_kl.detach()
    coeff = importance_ratio * (advantages - loss_config.kl_tau * log_importance_ratio)
    loss = -(coeff.detach() * trainer_logprobs)[keep_mask].sum()

    if loss_config.ratio_type == "sequence":
        loss = loss / torch.clamp_min(loss_mask.sum(), 1)

    metrics = {
        "mismatch_kl": _safe_mean(token_mismatch_kl, loss_mask),
        "masked_mismatch_kl": _safe_mean(token_mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(token_mismatch_kl, keep_mask),
        "is_masked": is_masked[loss_mask].float(),
        "is_masked_low": token_mask_low_mask[loss_mask].float(),
        "is_masked_high": token_mask_high_mask[loss_mask].float(),
        "sequence_masked_low": seq_mask_low.float(),
        "sequence_masked_high": seq_mask_high.float(),
        "geo_masked_low": geo_mask_low_mask.float(),
        "geo_masked_high": geo_mask_high_mask.float(),
        "geo_seq_ratio": geo_seq_ratio,
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)

    return LossOutputs(loss=loss, metrics=metrics)


def setup_loss_fn(loss_config: LossConfigType) -> LossFn:
    """Setup the loss function based on config."""
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def loss_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)

        return loss_fn

    def loss_fn(inputs: LossInputs) -> LossOutputs:
        return default_loss_fn(inputs, loss_config)

    return loss_fn


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fn: LossFn,
    loss_scale: int,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fn: Per-sequence loss function
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    total_loss = 0.0
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    for t_logp, i_logp, teach_logp, adv, mask in zip(
        trainer_logprobs, inference_logprobs, teacher_logprobs, advantages, loss_mask
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )

        result = loss_fn(inputs)

        total_loss = total_loss + result.loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    scaled_loss = total_loss / loss_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
