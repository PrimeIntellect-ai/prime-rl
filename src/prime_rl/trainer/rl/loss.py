import warnings
from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, LossConfig
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


def _compile_dynamic(fn: Callable) -> Callable:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script_method` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        return torch.compile(dynamic=True)(fn)


@jaxtyped(typechecker=typechecker)
@_compile_dynamic
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@_compile_dynamic
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


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def _kl_loss(log_importance_ratio: Tensor, loss_mask: Tensor, kl_level: str) -> tuple[Tensor, dict[str, Tensor]]:
    if kl_level == "token":
        return (loss_mask * log_importance_ratio**2).sum(), {}
    if kl_level == "sequence":
        sequence_log_importance_ratio = log_importance_ratio[loss_mask].sum()
        sequence_kl = sequence_log_importance_ratio**2
        return sequence_kl, {"sequence_kl": sequence_kl}
    raise ValueError(f"Invalid kl_level: {kl_level}")


def _clip_importance_ratio(raw_importance_ratio: Tensor, importance_ratio_clip: float | None) -> Tensor:
    if importance_ratio_clip is None:
        return raw_importance_ratio
    return raw_importance_ratio.clamp(max=importance_ratio_clip)


def _importance_ratio_metrics(
    raw_importance_ratio: Tensor, importance_ratio: Tensor, loss_mask: Tensor
) -> dict[str, Tensor]:
    return {
        "importance_ratio": _safe_mean(importance_ratio, loss_mask),
        "importance_ratio_raw": _safe_mean(raw_importance_ratio, loss_mask),
        "importance_ratio_clipped": _safe_mean(raw_importance_ratio > importance_ratio, loss_mask),
    }


def dppo_kl_loss_fn(
    inputs: LossInputs,
    dppo_mask_low: float = 0.2,
    dppo_mask_high: float = 0.2,
    adv_tau: float = 1.0,
    teacher_tau: float = 0.0,
    kl_tau: float = 1e-3,
    kl_level: str = "token",
    importance_ratio_clip: float | None = None,
) -> LossOutputs:
    """
    DPPO+KL loss, combining:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    The mask is conditioned on the advantage sign: for positive advantages,
    we mask tokens whose probability increased too much (trust region violation
    in the upweight direction); for negative advantages, we mask tokens whose
    probability decreased too much (trust region violation in the downweight
    direction).
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, raw_importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )
    importance_ratio = _clip_importance_ratio(raw_importance_ratio, importance_ratio_clip)

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        advantages = advantages + teacher_tau * teacher_kl.detach()
    else:
        teacher_kl = None

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss, kl_metrics = _kl_loss(log_importance_ratio, loss_mask, kl_level)
    loss = -pg_loss.sum() + kl_tau * kl_loss

    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        **_importance_ratio_metrics(raw_importance_ratio, importance_ratio, loss_mask),
        **kl_metrics,
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)

    return LossOutputs(loss=loss, metrics=metrics)


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    return dppo_kl_loss_fn(
        inputs,
        dppo_mask_low=loss_config.dppo_mask_low,
        dppo_mask_high=loss_config.dppo_mask_high,
        adv_tau=loss_config.adv_tau,
        teacher_tau=loss_config.teacher_tau,
        kl_tau=loss_config.kl_tau,
        importance_ratio_clip=loss_config.importance_ratio_clip,
    )


def opd_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    On-policy distillation loss: the default DPPO+KL math with the tau knobs
    hardcoded to drop the reward signal and use the teacher KL as the
    per-token policy-gradient signal.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    if teacher_logprobs is None:
        raise ValueError("opd_loss_fn requires teacher_logprobs - configure a teacher for opd mode.")

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > 0.2
    dppo_invalid_mask_low = probs_diff < -0.2
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    teacher_kl = teacher_logprobs - trainer_logprobs
    advantages = 0.0 * advantages + 1.0 * teacher_kl.detach()

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + 1e-3 * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        "teacher_kl": _safe_mean(teacher_kl, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def is_reinforce_loss_fn(
    inputs: LossInputs,
    adv_tau: float = 1.0,
    kl_tau: float = 0.0,
    kl_level: str = "token",
    importance_ratio_clip: float | None = None,
) -> LossOutputs:
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = adv_tau * inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, raw_importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )
    raw_importance_ratio = raw_importance_ratio.detach()
    importance_ratio = _clip_importance_ratio(raw_importance_ratio, importance_ratio_clip)
    kl_loss, kl_metrics = _kl_loss(log_importance_ratio, loss_mask, kl_level)

    pg_loss = loss_mask * advantages * importance_ratio * trainer_logprobs
    loss = -pg_loss.sum() + kl_tau * kl_loss
    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        **_importance_ratio_metrics(raw_importance_ratio, importance_ratio, loss_mask),
        **kl_metrics,
    }
    return LossOutputs(loss=loss, metrics=metrics)


def reinforce_loss_fn(
    inputs: LossInputs,
    adv_tau: float = 1.0,
    kl_tau: float = 0.0,
    kl_level: str = "token",
) -> LossOutputs:
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    rewards = adv_tau * inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )
    kl_loss, kl_metrics = _kl_loss(log_importance_ratio, loss_mask, kl_level)

    pg_loss = loss_mask * rewards * trainer_logprobs
    loss = -pg_loss.sum() + kl_tau * kl_loss
    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        **kl_metrics,
    }
    return LossOutputs(loss=loss, metrics=metrics)


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_loss_fns(loss_config: LossConfig) -> dict[str, LossFn]:
    """Build the per-training-mode loss fn dispatch table.

    Always returns all three modes - the trainer is mode-agnostic and routes
    per batch from ``TrainingSample.training_mode``:

    - ``"sft"`` → ``sft_loss_fn`` (masked NLL on teacher tokens)
    - ``"opd"`` → ``opd_loss_fn`` (teacher KL as gradient signal, hardcoded
      DPPO + KL knobs)
    - ``"rl"``  → ``default_loss_fn(loss_config)`` for ``DefaultLossConfig``,
      or the imported function for ``CustomLossConfig``.

    ``trainer.loss`` only affects the rl path - opd and sft are independent.
    """
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)
    else:

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    return {"sft": sft_loss_fn, "opd": opd_loss_fn, "rl": rl_fn}


def setup_loss_fn(loss_config: LossConfig) -> LossFn:
    """Compatibility wrapper for callers that still need a single RL loss fn."""
    return setup_loss_fns(loss_config)["rl"]


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn] | None = None,
    loss_scale: int | float = 1,
    training_mode: str = "rl",
    loss_fn: LossFn | None = None,
    sft_loss: bool = False,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Loss dispatch is batch-driven: ``training_mode`` selects the loss fn from
    ``loss_fns`` (built by ``setup_loss_fns``). sft → sft_loss_fn, opd →
    opd_loss_fn, rl → the configured default/custom loss.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fns: Per-mode loss fn dispatch table from setup_loss_fns()
        loss_scale: Scale factor to normalize the loss
        training_mode: Selects which loss fn to apply

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    if loss_fns is None:
        if loss_fn is None:
            raise ValueError("compute_loss requires either loss_fns or loss_fn.")
        effective_loss_fn = sft_loss_fn if sft_loss else loss_fn
    else:
        try:
            effective_loss_fn = loss_fns[training_mode]
        except KeyError:
            raise ValueError(
                f"No loss fn available for training_mode={training_mode!r} "
                f"(available: {sorted(loss_fns)}). Check trainer.loss.type."
            )

    total_loss = 0.0
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    for t_logp, i_logp, teach_logp, adv, mask in zip(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )

        result = effective_loss_fn(inputs)

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
