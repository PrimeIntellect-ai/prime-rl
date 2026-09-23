from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import (
    CustomLossConfig,
    DistributionalIPOLossConfig,
    IcePopLossConfig,
    IPOLossConfig,
    LossConfig,
    ScoreCenteringLossConfig,
)
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample.

    ``loss_mask`` already selects the tokens that belong to the receiving
    component — the component loss functions never re-derive eligibility.
    ``loss_weights`` is the component's per-token weight stream (None means
    1.0 everywhere).

    The ``topk_*`` fields carry the sampler's top-k head and the trainer's own
    gather at those ids, aligned per token like ``trainer_logprobs``; distributional
    IPO and score centering read them. ``entropy`` is the
    trainer policy's entropy (``plogp = -entropy``).
    """

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    ref_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]
    loss_weights: Float[Tensor, " seq"] | None = field(default=None)
    trainer_topk_logprobs: Float[Tensor, " seq k"] | None = field(default=None)
    sampler_topk_logprobs: Float[Tensor, " seq k"] | None = field(default=None)
    topk_valid: Bool[Tensor, " seq k"] | None = field(default=None)
    entropy: Float[Tensor, " seq"] | None = field(default=None)


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


class Loss(Protocol):
    """Interface for the config-initialized rl loss objects built by
    ``setup_rl_loss_fn``: ``IPOLoss``, ``IcePopLoss`` and ``CustomLoss``."""

    def loss(self, inputs: LossInputs) -> LossOutputs: ...


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function, as opposed to a ``Loss`` object: the
fixed ce / ref_kl losses and the function imported from ``CustomLossConfig``.

Expected signature for a custom loss:
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
def selective_topk_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], topk_ids: Int[Tensor, "batch seq k"]
) -> Float[Tensor, "batch seq k"]:
    """The trainer's logprobs at ``topk_ids``, full-vocab normalized.

    Padding ids (-1) yield 0.0 — consumers mask with the ids themselves. The
    logsumexp reduces over the vocab, so no full log-softmax materializes.
    """
    logz = torch.logsumexp(logits, dim=-1, keepdim=True)
    safe_ids = topk_ids.clamp_min(0).to(torch.int64)
    head_logprobs = logits.gather(dim=-1, index=safe_ids) - logz
    return head_logprobs.masked_fill(topk_ids < 0, 0.0)


@jaxtyped(typechecker=typechecker)
def selective_log_softmax_with_sampling_mask(
    logits: Float[Tensor, "batch seq vocab"],
    index: Int[Tensor, "batch seq"],
    sampling_mask: Int[Tensor, "batch seq mask"],
) -> Float[Tensor, "batch seq"]:
    """Per-token logprobs with sampling-mask replay: positions with a usable
    mask (see ``sampling_replay_mask``) get ``logits[index] -
    logsumexp(logits[mask])``, others full-vocab. Non-replayed rows are zeroed
    before the logsumexp so the unselected ``where`` branch can't emit NaN grads.
    """
    from prime_rl.trainer.models.layers.lm_head import sampling_replay_mask

    full_logprobs = selective_log_softmax(logits, index)
    replay = sampling_replay_mask(sampling_mask, index)
    mask_logits = torch.gather(logits, -1, sampling_mask.clamp_min(0).long())
    mask_logits = torch.where(sampling_mask >= 0, mask_logits, float("-inf"))
    mask_logits = torch.where(replay.unsqueeze(-1), mask_logits, 0.0)
    logz_masked = torch.logsumexp(mask_logits, dim=-1)
    target_logits = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(replay, target_logits - logz_masked, full_logprobs)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


def shift_tensor_left(t: Tensor, pad_value: float = 0.0) -> Tensor:
    """Shifts the tensor one position to the left along dim 1.

    Used to create labels from input_ids: labels[i] = input_ids[i+1]. The last
    position is padded with ``pad_value`` (0 is a valid token index but gets
    shifted off by shift_tensor_right and never used). Works for [batch, seq]
    labels and label-aligned [batch, seq, ...] fields like sampling_mask.
    """
    return torch.cat([t[:, 1:], torch.full_like(t[:, :1], pad_value)], dim=1)


def shift_tensor_right(
    t: Float[Tensor, "batch seq ..."], pad_value: float | None = None
) -> Float[Tensor, "batch seq ..."]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.
    Works for [batch, seq] and label-aligned [batch, seq, ...] fields like top-k heads.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    pad = torch.full((t.shape[0], 1, *t.shape[2:]), pad_value, device=t.device, dtype=t.dtype)
    return torch.cat([pad, t[:, :-1]], dim=1)


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


class IPOLoss:
    """IPO loss type: a symmetric trust region (mask tokens whose probability
    moved more than ``eps`` in absolute terms), policy gradient via
    the importance ratio, and a squared-log-ratio KL regularizer."""

    def __init__(self, config: IPOLossConfig):
        self.config = config

    def loss(self, inputs: LossInputs) -> LossOutputs:
        loss_config = self.config
        trainer_logprobs = inputs.trainer_logprobs
        inference_logprobs = inputs.inference_logprobs
        advantages = inputs.advantages
        loss_mask = inputs.loss_mask

        log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
            trainer_logprobs, inference_logprobs
        )

        abs_probs_diff = torch.abs(torch.exp(trainer_logprobs) - torch.exp(inference_logprobs))

        is_masked = abs_probs_diff > loss_config.eps
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


class DistributionalIPOLoss:
    """IPO with a total-variation gate and expected squared log ratio on candidates plus tail."""

    TAIL_EPS = 1e-8

    def __init__(self, config: DistributionalIPOLossConfig):
        self.config = config

    def loss(self, inputs: LossInputs) -> LossOutputs:
        if inputs.trainer_topk_logprobs is None or inputs.sampler_topk_logprobs is None or inputs.topk_valid is None:
            raise ValueError("distributional_ipo requires trainer and sampler candidate logprobs.")
        if (inputs.loss_mask & ~inputs.topk_valid.any(-1)).any():
            raise ValueError("distributional_ipo requires candidate logprobs at every RL member token.")

        valid = inputs.topk_valid & inputs.loss_mask.unsqueeze(-1)
        trainer_logp = inputs.trainer_topk_logprobs.masked_fill(~valid, 0.0)
        sampler_logp = inputs.sampler_topk_logprobs.detach().masked_fill(~valid, 0.0)
        trainer_probs = trainer_logp.exp().masked_fill(~valid, 0.0)
        sampler_probs = sampler_logp.exp().masked_fill(~valid, 0.0)
        trainer_tail = (1.0 - trainer_probs.sum(-1)).clamp_min(0.0)
        sampler_tail = (1.0 - sampler_probs.sum(-1)).clamp_min(0.0)

        # Coarsening the unrecorded tokens into one bucket hides movement within the tail.
        total_variation = 0.5 * (
            (trainer_probs.detach() - sampler_probs).abs().sum(-1) + (trainer_tail.detach() - sampler_tail).abs()
        )
        is_masked = total_variation > self.config.eps
        keep_mask = inputs.loss_mask & ~is_masked
        log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs.detach()
        safe_log_ratio = torch.where(keep_mask, log_ratio, 0.0)
        pg_loss = -(keep_mask * self.config.adv_tau * inputs.advantages.detach() * safe_log_ratio.exp())

        # The floor only stabilizes tail logs when subtraction rounds its mass to zero.
        tail_log_ratio = trainer_tail.clamp_min(self.TAIL_EPS).log() - sampler_tail.clamp_min(self.TAIL_EPS).log()
        squared_log_ratio = (sampler_probs * (trainer_logp - sampler_logp).square()).sum(-1)
        squared_log_ratio = squared_log_ratio + sampler_tail * tail_log_ratio.square()
        per_token_loss = pg_loss + self.config.kl_tau * squared_log_ratio
        if inputs.loss_weights is not None:
            per_token_loss = per_token_loss * inputs.loss_weights
        loss = per_token_loss[inputs.loss_mask].sum()

        with torch.no_grad():
            _, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
                inputs.trainer_logprobs, inputs.inference_logprobs
            )
        return LossOutputs(
            loss=loss,
            metrics={
                "is_masked": _safe_mean(is_masked, inputs.loss_mask),
                "masked_mismatch_kl": _safe_mean(mismatch_kl, inputs.loss_mask & is_masked),
                "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
                "total_variation": _safe_mean(total_variation, inputs.loss_mask),
                "squared_log_ratio": _safe_mean(squared_log_ratio.detach(), inputs.loss_mask),
                "head_mass": _safe_mean(sampler_probs.sum(-1), inputs.loss_mask),
            },
        )


class IcePopLoss:
    """IcePop loss type: policy gradient with a fixed importance-ratio
    acceptance band."""

    def __init__(self, config: IcePopLossConfig):
        self.config = config

    def loss(self, inputs: LossInputs) -> LossOutputs:
        loss_config = self.config
        log_importance_ratio, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
            inputs.trainer_logprobs, inputs.inference_logprobs
        )

        log_ratio_low = log_importance_ratio.new_tensor(loss_config.ratio_low).log()
        log_ratio_high = log_importance_ratio.new_tensor(loss_config.ratio_high).log()
        detached_log_ratio = log_importance_ratio.detach()
        is_masked = (detached_log_ratio < log_ratio_low) | (detached_log_ratio > log_ratio_high)
        keep_mask = inputs.loss_mask & ~is_masked

        # Mask before exponentiation so rejected extreme ratios cannot produce
        # 0 * inf = NaN in the loss or its gradient.
        safe_log_ratio = torch.where(keep_mask, log_importance_ratio, torch.zeros_like(log_importance_ratio))
        importance_ratio = torch.exp(safe_log_ratio)
        per_token_loss = -(keep_mask * loss_config.adv_tau * inputs.advantages * importance_ratio)
        if inputs.loss_weights is not None:
            per_token_loss = per_token_loss * inputs.loss_weights

        metrics = {
            "masked_mismatch_kl": _safe_mean(mismatch_kl, inputs.loss_mask & is_masked),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            "is_masked": _safe_mean(is_masked, inputs.loss_mask),
        }
        return LossOutputs(loss=per_token_loss.sum(), metrics=metrics)


class ScoreCenteringLoss:
    """Score centering loss ([arXiv:2609.20807](https://arxiv.org/abs/2609.20807)):
    policy gradient with a zero-expected-score baseline that cancels
    trainer/sampler drift on off-policy rollouts.

    The sampler's top-k head (ids + logprobs, recorded at rollout time) is
    transported to the trainer; the tail beyond it is modeled as proportional
    to the trainer's own distribution, rescaled to the sampler's tail mass
    (``q_tail = rho * p_tail``). Every ratio correction is constant over that
    modeled tail, so the backward pass touches only the k head gathers — the
    full-vocab entropy (``plogp``) enters detached.
    """

    # Floor on the modeled trainer tail mass; the paper's default.
    TAIL_EPS = 1e-6

    def __init__(self, config: ScoreCenteringLossConfig):
        self.config = config

    def loss(self, inputs: LossInputs) -> LossOutputs:
        if (
            inputs.trainer_topk_logprobs is None
            or inputs.sampler_topk_logprobs is None
            or inputs.topk_valid is None
            or inputs.entropy is None
        ):
            raise ValueError(
                "score_centering requires top-k sampler heads on every rl member token: request them "
                "with `logprobs = k` on the train sampling config (the rl entrypoint stamps it "
                "automatically) and run a vLLM >= 0.28 /inference/v1/generate server."
            )

        valid = inputs.topk_valid
        # The sampler's head q, and the trainer's gather at the same ids (the
        # only differentiable path). Padded columns read as zero mass.
        q_head = inputs.sampler_topk_logprobs.exp().masked_fill(~valid, 0.0)
        head_p = inputs.trainer_topk_logprobs
        p_head = head_p.exp().masked_fill(~valid, 0.0)
        # The tail's negative entropy enters as a constant (the paper stops its
        # gradient); the real pipeline computes it under no_grad, and detaching
        # keeps that invariant regardless of the caller.
        plogp = -inputs.entropy.detach()

        # The modeled tail: q_tail = alpha * p_tail, alpha detached throughout —
        # its gradient contribution is identically zero (E_p[grad log p] = 0).
        tail_q_mass = (1.0 - q_head.sum(-1)).clamp_min(0.0)
        p_tail = (1.0 - p_head.sum(-1)).clamp_min(self.TAIL_EPS)
        alpha = (tail_q_mass / p_tail).detach()
        residual = (q_head - alpha[..., None] * p_head).detach()

        # Zero-expected-score baseline: E_q[score] = 0, so the estimator stays
        # unbiased while the drift term cancels.
        center = (residual * head_p).sum(-1) + alpha * plogp
        score = inputs.trainer_logprobs - center

        per_token_loss = -inputs.advantages.detach() * score
        per_token_loss = per_token_loss * inputs.loss_mask
        if inputs.loss_weights is not None:
            per_token_loss = per_token_loss * inputs.loss_weights
        loss = per_token_loss.sum()

        _, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(inputs.trainer_logprobs, inputs.inference_logprobs)
        metrics = {
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, inputs.loss_mask),
            "head_mass": _safe_mean(q_head.sum(-1), inputs.loss_mask),
            "rho": _safe_mean(alpha, inputs.loss_mask),
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


class CustomLoss:
    """Custom loss type: the loss function imported from ``import_path``,
    called with ``kwargs``."""

    def __init__(self, config: CustomLossConfig):
        self.config = config
        self.fn: LossFn = import_object(config.import_path)

    def loss(self, inputs: LossInputs) -> LossOutputs:
        return self.fn(inputs, **self.config.kwargs)


def setup_rl_loss_fn(loss_config: LossConfig) -> Loss:
    """Build the loss object for the rl component from ``trainer.loss``.
    The ce / ref_kl loss types are fixed and unaffected by ``trainer.loss``."""
    match loss_config:
        case CustomLossConfig():
            return CustomLoss(loss_config)
        case DistributionalIPOLossConfig():
            return DistributionalIPOLoss(loss_config)
        case IPOLossConfig():
            return IPOLoss(loss_config)
        case IcePopLossConfig():
            return IcePopLoss(loss_config)
        case ScoreCenteringLossConfig():
            return ScoreCenteringLoss(loss_config)
        case _:
            raise TypeError(f"Unsupported RL loss config: {type(loss_config).__name__}")


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    ref_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    rl_weights: list[Float[Tensor, " seq_i"]] | None,
    ce_weights: list[Float[Tensor, " seq_i"]] | None,
    ref_kl_weights: list[Float[Tensor, " seq_i"]] | None,
    rl_loss_fn: Loss,
    rl_scale: int,
    ce_scale: int,
    ref_kl_scale: int,
    trainer_topk_logprobs: list[Float[Tensor, " seq_i k"]] | None = None,
    sampler_topk_logprobs: list[Float[Tensor, " seq_i k"]] | None = None,
    topk_valid: list[Bool[Tensor, " seq_i k"]] | None = None,
    entropy: list[Float[Tensor, " seq_i"]] | None = None,
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
        rl_loss_fn: RL loss object built by setup_rl_loss_fn()
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
    if trainer_topk_logprobs is None:
        trainer_topk_logprobs = [None] * n
    if sampler_topk_logprobs is None:
        sampler_topk_logprobs = [None] * n
    if topk_valid is None:
        topk_valid = [None] * n
    if entropy is None:
        entropy = [None] * n

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
    for t_logp, i_logp, ref_logp, adv, mask, rl_w, ce_w, ref_kl_w, t_topk, s_topk, topk_ok, ent in zip(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        trainer_topk_logprobs,
        sampler_topk_logprobs,
        topk_valid,
        entropy,
    ):

        def make_inputs(component_mask: Bool[Tensor, " seq"], weights: Float[Tensor, " seq"] | None) -> LossInputs:
            return LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                ref_logprobs=ref_logp,
                advantages=adv,
                loss_mask=component_mask,
                loss_weights=weights,
                trainer_topk_logprobs=t_topk,
                sampler_topk_logprobs=s_topk,
                topk_valid=topk_ok,
                entropy=ent,
            )

        if rl_w is None:
            rl_loss = rl_loss + run_loss_fn(rl_loss_fn.loss, make_inputs(mask, None))
        else:
            rl_mask = mask & (rl_w != 0)
            if bool(rl_mask.any()):
                rl_loss = rl_loss + run_loss_fn(rl_loss_fn.loss, make_inputs(rl_mask, rl_w))
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
