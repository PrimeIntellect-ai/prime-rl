from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import (
    CustomLossConfig,
    IcePopLossConfig,
    IPOLossConfig,
    LossConfig,
    ReinforceLossConfig,
    SamplerKLConfig,
)
from prime_rl.trainer.models.layers.lm_head import TokenSupport, support_replay_mask
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
    trainer_support_logprobs: Float[Tensor, " entries"] | None = None
    sampler_support_logprobs: Float[Tensor, " entries"] | None = None
    support_rows: Int[Tensor, " entries"] | None = None
    support_replayed: Bool[Tensor, " seq"] | None = None
    staleness: Int[Tensor, " seq"] | None = None


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


class Loss(Protocol):
    """Interface for the config-initialized rl loss objects built by
    ``setup_rl_loss_fn``."""

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


def _segment_logsumexp(values: Tensor, rows: Tensor, num_rows: int) -> Tensor:
    """Differentiable logsumexp per row of flat ``values``; -inf for empty rows (their gradient is zero)."""
    row_max = values.new_full((num_rows,), float("-inf")).scatter_reduce(0, rows, values.detach(), "amax")
    total = values.new_zeros(num_rows).index_add(0, rows, torch.exp(values - row_max[rows]))
    return row_max + torch.where(total > 0, total, 1.0).log()


def selective_log_softmax_with_support(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"], support: TokenSupport
) -> tuple[Float[Tensor, "batch seq"], Float[Tensor, " entries"]]:
    """Per-token logprobs with support replay, and the logprob of every support entry under the same
    normalization: tokens with a usable sampling mask (see ``support_replay_mask``) get ``logits[index] -
    logsumexp(logits[mask])``, others full-vocab."""
    logits, index = logits.squeeze(0), index.squeeze(0)
    rows = support.row_index().long()
    # Out-of-vocabulary ids (invalid evidence, raised after the step) get -inf like in the fused head
    in_vocab = (support.ids >= 0) & (support.ids < logits.shape[-1])
    support_logits = logits[rows, torch.where(in_vocab, support.ids, 0).long()].masked_fill(~in_vocab, float("-inf"))
    replay = support_replay_mask(support, index, rows)
    logz = torch.where(replay, _segment_logsumexp(support_logits, rows, support.num_rows), logits.logsumexp(-1))
    target_logits = logits.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return (target_logits - logz).unsqueeze(0), support_logits - logz[rows]


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
    labels and label-aligned [batch, seq, ...] fields.
    """
    return torch.cat([t[:, 1:], torch.full_like(t[:, :1], pad_value)], dim=1)


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
    return (values[mask] / denom).sum()


def _mismatch_kl_from_log_ratio(log_importance_ratio: Tensor) -> Tensor:
    # Keep headroom for FP32 reductions across tokens and ranks.
    metric_limit = log_importance_ratio.new_tensor(1e30)
    mismatch_kl = torch.expm1(log_importance_ratio.clamp(max=metric_limit.log())) - log_importance_ratio
    return mismatch_kl.clamp(max=metric_limit)


def _capped_importance_ratio(log_importance_ratio: Tensor, max_ratio: float) -> Tensor:
    capped_log_ratio = log_importance_ratio.detach().clamp(max=log_importance_ratio.new_tensor(max_ratio).log())
    return torch.exp(capped_log_ratio + (log_importance_ratio - log_importance_ratio.detach()))


@torch.no_grad()
def _importance_ratio_metrics(log_ratio: Tensor, keep_mask: Tensor) -> dict[str, Tensor]:
    """Per-token, pre-cap ratios for token-weighted reduction across samples and ranks.

    Ratio values saturate at 1e30 for finite moments; log ratios and exceedance
    indicators retain the tail information. Accepted means the sampled-action
    mask passes, including when SC also corrects rejected actions.
    """
    metrics = {}
    for population, values in (("all", log_ratio.detach()), ("accepted", log_ratio.detach()[keep_mask])):
        # TensorStats promotes to FP64 when accumulating moments.
        values = values.float()
        limit = values.new_tensor(1e30).log()
        metrics[f"importance_ratio/{population}"] = values.clamp(max=limit).exp()
        metrics[f"log_importance_ratio/{population}"] = values
        for threshold in (5, 10, 100, 10000):
            metrics[f"importance_ratio/{population}/above_{threshold}"] = values > values.new_tensor(threshold).log()
        metrics[f"importance_ratio/{population}/saturated"] = values > limit
    return metrics


def _sampler_support_distribution(
    inputs: LossInputs,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Trainer logprobs of every sampler support entry with weights ``u`` that estimate ``E_q[f]`` per token as
    ``sum_v u_v f(v)``: exact over the replayed support of truncated sampling (``u_v = q_v``), a K-sample estimate
    over K i.i.d. sampler draws of untruncated sampling (``u_v = 1/K``). Returns the support logprobs, the
    detached ``p``, ``log q`` and ``u``, and the per-token row sum over support entries."""
    if inputs.support_rows is None:
        raise ValueError("Score centering and the exact sampler KL need sampler support logprobs on every RL token")
    rows = inputs.support_rows.long()

    def row_sum(values: Tensor) -> Tensor:
        return values.new_zeros(inputs.loss_mask.shape[0]).index_add(0, rows, values)

    head = inputs.trainer_support_logprobs
    with torch.no_grad():
        p = head.exp()
        log_q = inputs.sampler_support_logprobs.detach()
        count = row_sum(torch.ones_like(log_q))
        if not bool((count > 0)[inputs.loss_mask].all()):
            raise ValueError("Score centering is missing sampler supports on RL tokens")
        u = torch.where(inputs.support_replayed[rows], log_q.exp(), 1.0 / count[rows].clamp_min(1.0))
    return head, p, log_q, u, row_sum


def _logit_correction_l1(mass: Tensor, p: Tensor, rows: Tensor, row_sum: Callable[[Tensor], Tensor]) -> Tensor:
    """Per-token L1 norm over the full vocabulary of ``sum_v mass_v * (e_v - p)``, the logit gradient of
    ``sum_v mass_v * log p_v``; the support-only ``p`` puts its remaining mass on the tail."""
    total_mass = row_sum(mass)
    return row_sum((mass - p * total_mass[rows]).abs()) + (1.0 - row_sum(p)).clamp_min(0.0) * total_mass.abs()


class SamplerKL:
    """KL(sampler || trainer) penalty on the rl tokens. Its gradient, ``-coef * E_q[grad log p]``,
    pulls the trainer toward the sampler independently of the advantage."""

    def __init__(self, config: SamplerKLConfig):
        self.config = config
        self.coef = config.coef

    def loss(self, inputs: LossInputs, weights: Tensor | None) -> tuple[Tensor, dict[str, Tensor]]:
        mask = inputs.loss_mask
        if self.config.min_staleness > 0:
            if inputs.staleness is None:
                raise ValueError("sampler_kl.min_staleness > 0 needs per-token staleness on the trainer")
            active = inputs.staleness[mask] >= self.config.min_staleness
        else:
            active = torch.ones_like(mask[mask])
        if self.config.estimator == "k1":
            trainer_logprobs = inputs.trainer_logprobs[mask]
            kl = (inputs.inference_logprobs[mask] - trainer_logprobs).detach()
            surrogate = -trainer_logprobs
        else:
            head, _, log_q, u, row_sum = _sampler_support_distribution(inputs)
            with torch.no_grad():
                # KL(q || p) = E_q[log q - log p], and its gradient is -E_q[grad log p].
                kl = row_sum(torch.where(u > 0, u * (log_q - head), 0.0))[mask]
            surrogate = -row_sum(u * head)[mask]
        per_token_loss = self.coef * surrogate * active
        if weights is not None:
            per_token_loss = per_token_loss * weights
        metrics = {
            "sampler_kl/kl": kl[active],
            "sampler_kl/k1": (inputs.inference_logprobs[mask] - inputs.trainer_logprobs[mask]).detach()[active],
            "sampler_kl/active": active.float(),
            "sampler_kl/coef": surrogate.new_tensor(self.coef),
        }
        return per_token_loss.sum(), metrics

    def adapt(self, kl: float) -> None:
        """Move the coefficient toward the adaptive target given the step's mean KL."""
        adaptive = self.config.adaptive
        if adaptive is None:
            return
        if kl > 1.5 * adaptive.target:
            self.coef *= 2.0
        elif kl < adaptive.target / 1.5:
            self.coef /= 2.0
        self.coef = min(max(self.coef, adaptive.min_coef), adaptive.max_coef)


class IPOLoss:
    """IPO loss type: a symmetric trust region (mask tokens whose probability
    moved more than ``eps`` in absolute terms), policy gradient via
    and a capped importance ratio."""

    def __init__(self, config: IPOLossConfig):
        self.config = config

    def loss(self, inputs: LossInputs) -> LossOutputs:
        loss_config = self.config
        trainer_logprobs = inputs.trainer_logprobs[inputs.loss_mask]
        inference_logprobs = inputs.inference_logprobs[inputs.loss_mask]
        advantages = inputs.advantages[inputs.loss_mask]
        weights = inputs.loss_weights[inputs.loss_mask] if inputs.loss_weights is not None else None

        log_importance_ratio = trainer_logprobs - inference_logprobs
        larger_logprob = torch.maximum(trainer_logprobs, inference_logprobs)
        smaller_logprob = torch.minimum(trainer_logprobs, inference_logprobs)
        # |e^logp - e^logq| = e^max(logp, logq) - e^min(logp, logq)
        # = e^max(logp, logq) * (1 - e^(min(logp, logq) - max(logp, logq)))
        # = e^max(logp, logq) * -expm1(min(logp, logq) - max(logp, logq)).
        # expm1 avoids cancellation in 1 - e^x when x is near zero.
        abs_probs_diff = torch.exp(larger_logprob) * -torch.expm1(smaller_logprob - larger_logprob)
        is_masked = abs_probs_diff > loss_config.eps
        keep_mask = ~is_masked

        importance_ratio = _capped_importance_ratio(log_importance_ratio[keep_mask], loss_config.max_importance_ratio)
        pg_loss = -loss_config.adv_tau * advantages[keep_mask] * importance_ratio
        if weights is not None:
            pg_loss = pg_loss * weights[keep_mask]
        loss = pg_loss.sum()

        mismatch_kl = _mismatch_kl_from_log_ratio(log_importance_ratio)

        metrics = {
            "masked_mismatch_kl": _safe_mean(mismatch_kl, is_masked),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            **_importance_ratio_metrics(log_importance_ratio, keep_mask),
            "is_masked": is_masked.sum() / max(is_masked.numel(), 1),
        }

        return LossOutputs(loss=loss, metrics=metrics)


class IcePopLoss:
    """IcePop loss type: policy gradient with a fixed importance-ratio
    acceptance band."""

    def __init__(self, config: IcePopLossConfig):
        self.config = config
        self.sampler_kl = SamplerKL(config.sampler_kl) if config.sampler_kl is not None else None

    def loss(self, inputs: LossInputs) -> LossOutputs:
        loss_config = self.config
        log_importance_ratio = inputs.trainer_logprobs[inputs.loss_mask] - inputs.inference_logprobs[inputs.loss_mask]
        advantages = inputs.advantages[inputs.loss_mask]
        weights = inputs.loss_weights[inputs.loss_mask] if inputs.loss_weights is not None else None

        log_ratio_low = log_importance_ratio.new_tensor(loss_config.ratio_low).log()
        log_ratio_high = log_importance_ratio.new_tensor(loss_config.ratio_high).log()
        detached_log_ratio = log_importance_ratio.detach()
        is_masked = (detached_log_ratio < log_ratio_low) | (detached_log_ratio > log_ratio_high)
        keep_mask = ~is_masked

        importance_ratio = torch.exp(log_importance_ratio[keep_mask])
        per_token_loss = -loss_config.adv_tau * advantages[keep_mask] * importance_ratio
        if weights is not None:
            per_token_loss = per_token_loss * weights[keep_mask]

        loss = per_token_loss.sum()
        centering_metrics = {}
        if loss_config.score_centering and log_importance_ratio.numel():
            mask = inputs.loss_mask
            head, p, log_q, u, row_sum = _sampler_support_distribution(inputs)
            rows = inputs.support_rows.long()
            with torch.no_grad():
                head_log_ratio = head - log_q
                accepted = (head_log_ratio >= log_ratio_low) & (head_log_ratio <= log_ratio_high)
                # E_q[(p/q) * accepted * grad log p] ~ sum_v u_v (p_v / q_v) accepted_v grad log p_v
                ratio = head_log_ratio.exp()
                mass = u * ratio * accepted
                centering_metrics["score_centering/replayed"] = inputs.support_replayed[mask].to(p.dtype)
                centering_metrics["score_centering/support_mass"] = row_sum(u * ratio)[mask]
                centering_metrics["score_centering/logit_correction_l1"] = _logit_correction_l1(mass, p, rows, row_sum)[
                    mask
                ]
            correction = loss_config.adv_tau * advantages.detach() * row_sum(mass * head)[mask]
            if weights is not None:
                correction = correction * weights
            # Rejected sampled actions still contribute the centering correction.
            loss = loss + correction.sum()
        if self.sampler_kl is not None and log_importance_ratio.numel():
            kl_loss, kl_metrics = self.sampler_kl.loss(inputs, weights)
            loss = loss + kl_loss
            centering_metrics.update(kl_metrics)

        mismatch_kl = _mismatch_kl_from_log_ratio(log_importance_ratio)

        metrics = {
            "masked_mismatch_kl": _safe_mean(mismatch_kl, is_masked),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            **_importance_ratio_metrics(log_importance_ratio, keep_mask),
            "is_masked": is_masked.sum() / max(is_masked.numel(), 1),
            **centering_metrics,
        }
        return LossOutputs(loss=loss, metrics=metrics)


class ReinforceLoss:
    """Reinforce loss type: policy gradient without importance weights, optionally centered
    under the sampler and regularized toward it."""

    def __init__(self, config: ReinforceLossConfig):
        self.config = config
        self.sampler_kl = SamplerKL(config.sampler_kl) if config.sampler_kl is not None else None

    def loss(self, inputs: LossInputs) -> LossOutputs:
        loss_config = self.config
        trainer_logprobs = inputs.trainer_logprobs[inputs.loss_mask]
        advantages = inputs.advantages[inputs.loss_mask].detach()
        weights = inputs.loss_weights[inputs.loss_mask] if inputs.loss_weights is not None else None

        per_token_loss = -loss_config.adv_tau * advantages * trainer_logprobs
        metrics = {}
        if loss_config.score_centering and trainer_logprobs.numel():
            head, p, log_q, u, row_sum = _sampler_support_distribution(inputs)
            rows = inputs.support_rows.long()
            with torch.no_grad():
                # E_q[grad log p] ~ sum_v u_v grad log p_v
                metrics["score_centering/replayed"] = inputs.support_replayed[inputs.loss_mask].to(p.dtype)
                metrics["score_centering/support_mass"] = row_sum(u * (head - log_q).exp())[inputs.loss_mask]
                metrics["score_centering/logit_correction_l1"] = _logit_correction_l1(u, p, rows, row_sum)[
                    inputs.loss_mask
                ]
            per_token_loss = per_token_loss + loss_config.adv_tau * advantages * row_sum(u * head)[inputs.loss_mask]
        if weights is not None:
            per_token_loss = per_token_loss * weights
        loss = per_token_loss.sum()

        if self.sampler_kl is not None and trainer_logprobs.numel():
            kl_loss, kl_metrics = self.sampler_kl.loss(inputs, weights)
            loss = loss + kl_loss
            metrics.update(kl_metrics)
        return LossOutputs(loss=loss, metrics=metrics)


def ref_kl_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    Ref-KL loss type (on-policy distillation): the reverse KL to the reference
    model is the per-token policy-gradient signal, with the importance ratio
    correcting trainer/inference mismatch and staleness. A one-sided trust
    region drops tokens whose trainer probability fell more than 0.2 below the
    inference probability. Scalar
    advantages are not read — ref_kl algorithms ship none.
    """
    if inputs.ref_logprobs is None:
        raise ValueError("ref_kl loss type requires ref_logprobs — use the 'opd' or 'opsd' algorithm.")

    trainer_logprobs = inputs.trainer_logprobs[inputs.loss_mask]
    inference_logprobs = inputs.inference_logprobs[inputs.loss_mask]
    ref_logprobs = inputs.ref_logprobs[inputs.loss_mask]
    weights = inputs.loss_weights[inputs.loss_mask] if inputs.loss_weights is not None else None
    log_importance_ratio = trainer_logprobs - inference_logprobs

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    is_masked = probs_diff < -0.2
    keep_mask = ~is_masked

    ref_kl = ref_logprobs - trainer_logprobs

    importance_ratio = _capped_importance_ratio(log_importance_ratio[keep_mask], 1e4)
    pg_loss = -ref_kl[keep_mask].detach() * importance_ratio
    if weights is not None:
        pg_loss = pg_loss * weights[keep_mask]
    loss = pg_loss.sum()
    mismatch_kl = _mismatch_kl_from_log_ratio(log_importance_ratio)

    # Namespaced: the rl loss fn emits same-named trust-region metrics with a
    # different definition, and mixed batches run both fns in one step.
    metrics = {
        "ref_kl/masked_mismatch_kl": _safe_mean(mismatch_kl, is_masked),
        "ref_kl/unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "ref_kl/is_masked": is_masked.sum() / max(is_masked.numel(), 1),
        "ref_kl": ref_kl.sum() / max(ref_kl.numel(), 1),
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
        case IPOLossConfig():
            return IPOLoss(loss_config)
        case IcePopLossConfig():
            return IcePopLoss(loss_config)
        case ReinforceLossConfig():
            return ReinforceLoss(loss_config)
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
    rl_scale: float,
    ce_scale: float,
    ref_kl_scale: float,
    support: TokenSupport | None = None,
    trainer_support_logprobs: Float[Tensor, " entries"] | None = None,
    staleness: list[Int[Tensor, " seq_i"]] | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    The loss is a sum of three components, each running over its own per-token
    weight stream and normalized by its own global denominator (rl: sum of its
    weights; ce / ref_kl: token count):

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
        rl_scale: Divisor of the rl loss sum
        ce_scale: Divisor of the ce loss sum
        ref_kl_scale: Divisor of the ref_kl loss sum
        support: Token-aligned sampler support of the whole micro batch, or None
        trainer_support_logprobs: Trainer logprobs of every ``support`` entry
        staleness: Per-token staleness for each sequence, or None

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
    if staleness is None:
        staleness = [None] * n

    sample_supports = [{}] * n
    if trainer_support_logprobs is not None:
        rows = support.row_index()
        sample_supports, start = [], 0
        for logprobs in trainer_logprobs:
            end = start + logprobs.shape[0]
            first, last = int(support.offsets[start]), int(support.offsets[end])
            sample_supports.append(
                dict(
                    trainer_support_logprobs=trainer_support_logprobs[first:last],
                    sampler_support_logprobs=support.logprobs[first:last],
                    support_rows=rows[first:last] - start,
                    support_replayed=support.replayed[start:end],
                )
            )
            start = end

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
    for t_logp, i_logp, ref_logp, adv, mask, rl_w, ce_w, ref_kl_w, sample_support, stale in zip(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        sample_supports,
        staleness,
        strict=True,
    ):

        def make_inputs(component_mask: Bool[Tensor, " seq"], weights: Float[Tensor, " seq"] | None) -> LossInputs:
            return LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                ref_logprobs=ref_logp,
                advantages=adv,
                loss_mask=component_mask,
                loss_weights=weights,
                staleness=stale,
                **sample_support,
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
