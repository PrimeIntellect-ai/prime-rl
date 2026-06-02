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
    # Echo tokens are excluded from RL loss/metrics and trained through the
    # echo CE term. The advantage tensor carries alpha on these positions.
    echo_mask: Bool[Tensor, " seq"] | None = None
    rl_loss_scale: int = 1
    echo_loss_scale: int = 1


@dataclass
class LossMasks:
    loss: Bool[Tensor, " seq"]
    rl: Bool[Tensor, " seq"]
    echo: Bool[Tensor, " seq"]


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


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def split_loss_masks(loss_mask: Tensor, echo_mask: Tensor | None) -> LossMasks:
    if echo_mask is None:
        echo_train_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        rl_mask = loss_mask
    else:
        echo_train_mask = loss_mask & echo_mask
        rl_mask = loss_mask & ~echo_mask
    return LossMasks(loss=loss_mask, rl=rl_mask, echo=echo_train_mask)


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    DPPO+KL loss for RL training, combining:
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
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask
    masks = split_loss_masks(loss_mask, inputs.echo_mask)

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    dppo_drop_mask = masks.rl & dppo_invalid_mask
    dppo_keep_mask = masks.rl & ~dppo_invalid_mask
    is_masked_high = masks.rl & positive_advantages & dppo_invalid_mask_high
    is_masked_low = masks.rl & negative_advantages & dppo_invalid_mask_low

    advantages = loss_config.adv_tau * advantages
    rl_pg_loss = dppo_keep_mask * advantages * importance_ratio
    rl_kl_loss = masks.rl * log_importance_ratio**2
    rl_loss = (-rl_pg_loss + loss_config.kl_tau * rl_kl_loss).sum() / inputs.rl_loss_scale

    if inputs.echo_mask is not None and masks.echo.any():
        echo_loss = -(advantages * trainer_logprobs)[masks.echo].sum() / inputs.echo_loss_scale
    else:
        echo_loss = torch.zeros((), device=trainer_logprobs.device, dtype=trainer_logprobs.dtype)

    loss = rl_loss + echo_loss

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, dppo_drop_mask),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, dppo_keep_mask),
        "is_masked": _safe_mean(dppo_drop_mask, masks.rl),
        "is_masked_low": _safe_mean(is_masked_low, masks.rl),
        "is_masked_high": _safe_mean(is_masked_high, masks.rl),
        "masked_advantage_positive": _safe_mean(positive_advantages, dppo_drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, dppo_drop_mask),
    }

    if inputs.echo_mask is not None:
        metrics["echo_nll_mean"] = _safe_mean(-trainer_logprobs, masks.echo)
        metrics["echo_token_count"] = masks.echo.sum().float()
        if masks.echo.any():
            metrics["echo_nll_max"] = (-trainer_logprobs[masks.echo]).max()
        else:
            metrics["echo_nll_max"] = torch.tensor(0.0, device=trainer_logprobs.device)

    return LossOutputs(loss=loss, metrics=metrics)


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
    loss = (-pg_loss + 1e-3 * kl_loss).sum() / inputs.rl_loss_scale

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


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum() / inputs.rl_loss_scale
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
            if inputs.echo_mask is not None and inputs.echo_mask.any():
                raise ValueError(
                    "Echo is only supported with the default RL loss. "
                    "CustomLossConfig receives the legacy loss_mask/advantages contract and cannot safely interpret echo."
                )
            result = custom_fn(inputs, **kwargs)
            return LossOutputs(loss=result.loss / inputs.rl_loss_scale, metrics=result.metrics)
    else:

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    return {"sft": sft_loss_fn, "opd": opd_loss_fn, "rl": rl_fn}


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn],
    rl_loss_scale: int,
    training_mode: str = "rl",
    echo_mask: list[Bool[Tensor, " seq_i"]] | None = None,
    echo_loss_scale: int | None = None,
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
        rl_loss_scale: Global RL/non-echo token denominator
        training_mode: Selects which loss fn to apply
        echo_mask: Per-sequence echo masks (parallel to loss_mask). Echo tokens
            are excluded from RL terms and trained through the echo CE term.
        echo_loss_scale: Global echo token denominator. Defaults to rl_loss_scale
            for backward-compatible direct calls.

    Returns:
        Tuple of (total_loss, aggregated_metrics)
    """
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
    if echo_mask is None:
        echo_mask_list: list[Bool[Tensor, " seq_i"] | None] = [None] * len(trainer_logprobs)
    else:
        echo_mask_list = list(echo_mask)
    if echo_loss_scale is None:
        echo_loss_scale = rl_loss_scale

    for t_logp, i_logp, teach_logp, adv, mask, echo_m in zip(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
        echo_mask_list,
        strict=True,
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
            echo_mask=echo_m,
            rl_loss_scale=rl_loss_scale,
            echo_loss_scale=echo_loss_scale,
        )
        if echo_m is not None and echo_m.any() and training_mode != "rl":
            raise ValueError("Echo is only supported for training_mode='rl'.")

        result = effective_loss_fn(inputs)

        total_loss = total_loss + result.loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return total_loss, aggregated
