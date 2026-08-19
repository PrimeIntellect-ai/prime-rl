from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, IPOLossConfig, LossConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for one loss-component call.

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
    """Outputs from one loss-component call."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-component loss function.

``compute_loss`` calls it once per micro batch, over the packed sequences
concatenated along the token dimension. All built-in loss fns are per-token
sums, for which the concatenation is equivalent to per-sequence calls.

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


@dataclass(frozen=True)
class PGParams:
    """One instance of the policy-gradient skeleton (see ``pg_loss_fn``).

    The default (DPPO), IPO, and ref_kl loss types are the same per-token loss

        -(keep_mask · adv_tau·A · importance_ratio) + kl_tau · loss_mask · log_ratio²

    differing only in where the advantage A comes from, which trust region
    gates ``keep_mask``, and the drift coefficient ``kl_tau``.
    """

    advantage: Literal["shipped", "ref_kl"]
    trust_region: Literal["dppo", "ipo", "one_sided"]
    adv_tau: float
    kl_tau: float
    dppo_mask_low: float = 0.0
    dppo_mask_high: float = 0.0
    ipo_threshold: float = 0.0
    one_sided_threshold: float = 0.2
    metric_prefix: str = ""


def pg_loss_fn(inputs: LossInputs, params: PGParams) -> LossOutputs:
    """The shared policy-gradient skeleton behind the rl and ref_kl loss types.

    Advantage providers:
    - "shipped": the orchestrator-stamped advantage stream (data, fixed at
      rollout time).
    - "ref_kl": ``(ref_logprobs - trainer_logprobs).detach()`` — the reverse KL
      to the reference model is the per-token policy-gradient signal
      (∇KL(π_θ‖π_ref) = -E[(log π_ref - log π_θ)·∇log π_θ]), recomputed every
      forward so the pull anneals as the policy approaches the reference.

    Trust regions (on ``probs_diff = p_trainer - p_inference``):
    - "dppo": conditioned on the advantage sign — for positive advantages, mask
      tokens whose probability increased too much; for negative advantages,
      tokens whose probability decreased too much. DPPO-Binary TV Loss
      (https://arxiv.org/pdf/2602.04879) + Kimi-K2.5 KL Loss
      (https://arxiv.org/pdf/2602.02276).
    - "ipo": symmetric — mask tokens whose probability moved more than
      ``ipo_threshold`` in absolute terms.
    - "one_sided": mask tokens whose trainer probability already fell more than
      ``one_sided_threshold`` below the inference probability.

    In all instances the importance ratio corrects trainer/inference mismatch
    and staleness, and the squared-log-ratio term regularizes drift.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    if params.advantage == "shipped":
        advantages = inputs.advantages
    else:
        if inputs.ref_logprobs is None:
            raise ValueError("ref_kl loss type requires ref_logprobs — use the 'opd' or 'opsd' algorithm.")
        advantages = (inputs.ref_logprobs - trainer_logprobs).detach()

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    if params.trust_region == "dppo":
        positive_advantages = advantages > 0
        negative_advantages = advantages < 0
        dppo_invalid_mask_high = probs_diff > params.dppo_mask_high
        dppo_invalid_mask_low = probs_diff < -params.dppo_mask_low
        is_masked = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)
    elif params.trust_region == "ipo":
        is_masked = torch.abs(probs_diff) > params.ipo_threshold
    else:  # one_sided
        is_masked = probs_diff < -params.one_sided_threshold

    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    pg_loss = keep_mask * (params.adv_tau * advantages) * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + params.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    # Prefixed per instance: rl and ref_kl emit same-named trust-region metrics
    # with different definitions, and mixed batches run both in one step.
    prefix = params.metric_prefix
    metrics = {
        f"{prefix}masked_mismatch_kl": _safe_mean(mismatch_kl, drop_mask),
        f"{prefix}unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        f"{prefix}is_masked": _safe_mean(is_masked, loss_mask),
    }
    if params.trust_region == "dppo":
        metrics[f"{prefix}is_masked_high"] = _safe_mean(positive_advantages & dppo_invalid_mask_high, loss_mask)
        metrics[f"{prefix}is_masked_low"] = _safe_mean(negative_advantages & dppo_invalid_mask_low, loss_mask)
        metrics[f"{prefix}masked_advantage_positive"] = _safe_mean(positive_advantages, drop_mask)
        metrics[f"{prefix}masked_advantage_negative"] = _safe_mean(negative_advantages, drop_mask)
    if params.advantage == "ref_kl":
        metrics[f"{prefix.rstrip('/')}"] = _safe_mean(advantages, loss_mask)

    return LossOutputs(loss=loss, metrics=metrics)


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """DPPO+KL loss for RL training: the pg skeleton with shipped advantages
    and the sign-conditioned trust region."""
    params = PGParams(
        advantage="shipped",
        trust_region="dppo",
        adv_tau=loss_config.adv_tau,
        kl_tau=loss_config.kl_tau,
        dppo_mask_low=loss_config.dppo_mask_low,
        dppo_mask_high=loss_config.dppo_mask_high,
    )
    return pg_loss_fn(inputs, params)


def ipo_loss_fn(inputs: LossInputs, loss_config: IPOLossConfig) -> LossOutputs:
    """IPO loss type: the pg skeleton with shipped advantages and a symmetric
    trust region."""
    params = PGParams(
        advantage="shipped",
        trust_region="ipo",
        adv_tau=loss_config.adv_tau,
        kl_tau=loss_config.kl_tau,
        ipo_threshold=loss_config.ipo_threshold,
    )
    return pg_loss_fn(inputs, params)


_REF_KL_PARAMS = PGParams(
    advantage="ref_kl",
    trust_region="one_sided",
    adv_tau=1.0,
    kl_tau=1e-3,
    one_sided_threshold=0.2,
    metric_prefix="ref_kl/",
)


def ref_kl_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Ref-KL loss type (on-policy distillation): the pg skeleton with the
    reverse KL to the reference model as the advantage and a one-sided trust
    region. Scalar advantages are not read — ref_kl algorithms ship none."""
    return pg_loss_fn(inputs, _REF_KL_PARAMS)


def ce_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Cross-entropy loss type: masked negative log-likelihood (SFT / ECHO
    observation prediction). The one component outside the pg skeleton: ce
    tokens (SFT data, observations, hint blocks) were never sampled from the
    policy, so an importance ratio on them is meaningless."""
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
    weight stream and normalized by its own global token count. Every component
    runs ONCE over the packed sequences concatenated along the token dimension —
    there is no per-sequence loop and no device-side presence check; a component
    whose mask is empty contributes an exact zero (and zero-valued metrics).

    - rl → ``rl_loss_fn`` (built by ``setup_rl_loss_fn``) on
      ``loss_mask & (rl_weights != 0)``; an absent stream means weight 1.0 on
      the full loss mask (the hot path). The rl component always runs, so the
      returned loss is always backward-able and every rank runs backward even
      when all components are empty (e.g. a fully truncated distillation
      sample, whose stamped streams survive as all-zero prefixes) — FSDP
      collectives stay in sync.
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
    trainer_cat = torch.cat(trainer_logprobs)
    inference_cat = torch.cat(inference_logprobs)
    ref_cat = torch.cat(ref_logprobs) if ref_logprobs is not None else None
    advantages_cat = torch.cat(advantages)
    mask_cat = torch.cat(loss_mask)
    rl_w = torch.cat(rl_weights) if rl_weights is not None else None
    ce_w = torch.cat(ce_weights) if ce_weights is not None else None
    ref_kl_w = torch.cat(ref_kl_weights) if ref_kl_weights is not None else None

    metrics: dict[str, Any] = {}

    def run_loss_fn(
        loss_fn: LossFn, component_mask: Bool[Tensor, " seq"], weights: Float[Tensor, " seq"] | None
    ) -> Tensor:
        result = loss_fn(
            LossInputs(
                trainer_logprobs=trainer_cat,
                inference_logprobs=inference_cat,
                ref_logprobs=ref_cat,
                advantages=advantages_cat,
                loss_mask=component_mask,
                loss_weights=weights,
            )
        )
        for k, v in result.metrics.items():
            metrics[k] = v.unsqueeze(0) if v.dim() == 0 else v
        return result.loss

    if rl_w is None:
        rl_loss = run_loss_fn(rl_loss_fn, mask_cat, None)
    else:
        rl_loss = run_loss_fn(rl_loss_fn, mask_cat & (rl_w != 0), rl_w)
    ce_loss = run_loss_fn(ce_loss_fn, ce_w != 0, ce_w) if ce_w is not None else 0.0
    ref_kl_loss = run_loss_fn(ref_kl_loss_fn, ref_kl_w != 0, ref_kl_w) if ref_kl_w is not None else 0.0

    scaled_loss = rl_loss / rl_scale + ce_loss / ce_scale + ref_kl_loss / ref_kl_scale
    return scaled_loss, metrics
