from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, IPOLossConfig, LossConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample.

    ``loss_mask`` already selects the tokens that belong to the receiving
    component — the component loss functions never re-derive eligibility.
    ``loss_weights`` is the component's per-token weight stream (None means
    1.0 everywhere).
    """

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    ref_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]
    loss_weights: Float[Tensor, " seq"] | None = field(default=None)


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

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ipo_loss_fn(inputs: LossInputs, loss_config: IPOLossConfig) -> LossOutputs:
    """IPO loss type: a symmetric trust region (mask tokens whose probability
    moved more than ``ipo_threshold`` in absolute terms), policy gradient via
    the importance ratio, and a squared-log-ratio KL regularizer."""
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    abs_probs_diff = torch.abs(torch.exp(trainer_logprobs) - torch.exp(inference_logprobs))

    is_masked = abs_probs_diff > loss_config.ipo_threshold
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ref_kl_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    Ref-KL loss type (on-policy distillation): the reverse KL to the reference
    model is the per-token policy-gradient signal, with the importance ratio
    correcting trainer/inference mismatch and staleness. A one-sided trust
    region drops tokens whose trainer probability fell more than 0.2 below the
    inference probability; a squared-log-ratio term regularizes drift. Scalar
    advantages are not read — ref_kl algorithms ship none.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    ref_logprobs = inputs.ref_logprobs
    loss_mask = inputs.loss_mask

    if ref_logprobs is None:
        raise ValueError("ref_kl loss type requires ref_logprobs — use the 'opd' or 'opsd' algorithm.")

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    is_masked = probs_diff < -0.2
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    ref_kl = ref_logprobs - trainer_logprobs

    pg_loss = keep_mask * ref_kl.detach() * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + 1e-3 * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    # Namespaced: the rl loss fn emits same-named trust-region metrics with a
    # different definition, and mixed batches run both fns in one step.
    metrics = {
        "ref_kl/masked_mismatch_kl": _safe_mean(mismatch_kl, drop_mask),
        "ref_kl/unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "ref_kl/is_masked": _safe_mean(is_masked, loss_mask),
        "ref_kl": _safe_mean(ref_kl, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ce_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Cross-entropy loss type: masked negative log-likelihood (SFT / ECHO
    observation prediction)."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    nll = -trainer_logprobs
    if inputs.loss_weights is not None:
        nll = nll * inputs.loss_weights
    loss = nll[loss_mask].sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_rl_loss_fn(loss_config: LossConfig) -> LossFn:
    """Build the loss fn for the rl component from ``trainer.loss``:
    ``default_loss_fn`` (``DefaultLossConfig``), ``ipo_loss_fn``
    (``IPOLossConfig``), or the imported function (``CustomLossConfig``).
    The ce / ref_kl loss types are fixed and unaffected by ``trainer.loss``."""
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)
    elif isinstance(loss_config, IPOLossConfig):

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return ipo_loss_fn(inputs, loss_config)
    else:

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    return rl_fn


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    ref_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    rl_weights: list[Float[Tensor, " seq_i"]] | None,
    ce_weights: list[Float[Tensor, " seq_i"]] | None,
    ref_kl_weights: list[Float[Tensor, " seq_i"]] | None,
    rl_loss_fn: LossFn,
    rl_scale: int,
    ce_scale: int,
    ref_kl_scale: int,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    The loss is a sum of three components, each running over its own per-token
    weight stream and normalized by its own global token count:

    - rl → ``rl_loss_fn`` (built by ``setup_rl_loss_fn``) on
      ``loss_mask & (rl_weights != 0)``; an absent stream means weight 1.0 on
      the full loss mask (the hot path — no extra device syncs).
    - ce → ``ce_loss_fn`` (masked NLL) on ``ce_weights != 0``.
    - ref_kl → ``ref_kl_loss_fn`` on ``ref_kl_weights != 0``.

    A weight scales its component's per-token loss; 0.0 removes the token from
    the component's mask and denominator. Per-component normalization keeps the
    components from diluting each other: a token only enters the denominator of
    the components it belongs to.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Sampling-policy log probabilities for each sequence
        ref_logprobs: Reference-model log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        rl_weights: Per-token rl weights for each sequence, or None (1.0 on the loss mask)
        ce_weights: Per-token ce weights for each sequence, or None (no ce component)
        ref_kl_weights: Per-token ref_kl weights for each sequence, or None (no ref_kl component)
        rl_loss_fn: Loss fn for the rl component from setup_rl_loss_fn()
        rl_scale: Global rl-token count normalizing the rl component
        ce_scale: Global ce-token count normalizing the ce component
        ref_kl_scale: Global ref_kl-token count normalizing the ref_kl component

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    all_metrics: dict[str, list[Tensor]] = {}

    n = len(trainer_logprobs)
    if ref_logprobs is None:
        ref_logprobs = [None] * n
    if rl_weights is None:
        rl_weights = [None] * n
    if ce_weights is None:
        ce_weights = [None] * n
    if ref_kl_weights is None:
        ref_kl_weights = [None] * n

    def run_loss_fn(loss_fn: LossFn, inputs: LossInputs) -> Tensor:
        result = loss_fn(inputs)
        for k, v in result.metrics.items():
            all_metrics.setdefault(k, []).append(v)
        return result.loss

    # Graph anchor: a micro batch whose components are all empty (e.g. a fully
    # truncated distillation sample, whose stamped streams survive as all-zero
    # prefixes) must still return a backward-able loss so every rank runs
    # backward and FSDP collectives stay in sync.
    rl_loss = trainer_logprobs[0].sum() * 0.0
    ce_loss = 0.0
    ref_kl_loss = 0.0
    for t_logp, i_logp, ref_logp, adv, mask, rl_w, ce_w, ref_kl_w in zip(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask,
        rl_weights,
        ce_weights,
        ref_kl_weights,
    ):

        def make_inputs(component_mask: Bool[Tensor, " seq"], weights: Float[Tensor, " seq"] | None) -> LossInputs:
            return LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                ref_logprobs=ref_logp,
                advantages=adv,
                loss_mask=component_mask,
                loss_weights=weights,
            )

        if rl_w is None:
            rl_loss = rl_loss + run_loss_fn(rl_loss_fn, make_inputs(mask, None))
        else:
            rl_mask = mask & (rl_w != 0)
            if bool(rl_mask.any()):
                rl_loss = rl_loss + run_loss_fn(rl_loss_fn, make_inputs(rl_mask, rl_w))
        if ce_w is not None:
            ce_mask = ce_w != 0
            if bool(ce_mask.any()):
                ce_loss = ce_loss + run_loss_fn(ce_loss_fn, make_inputs(ce_mask, ce_w))
        if ref_kl_w is not None:
            ref_kl_mask = ref_kl_w != 0
            if bool(ref_kl_mask.any()):
                ref_kl_loss = ref_kl_loss + run_loss_fn(ref_kl_loss_fn, make_inputs(ref_kl_mask, ref_kl_w))

    scaled_loss = rl_loss / rl_scale + ce_loss / ce_scale + ref_kl_loss / ref_kl_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
