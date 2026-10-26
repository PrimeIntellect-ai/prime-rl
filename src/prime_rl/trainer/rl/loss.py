import math
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
    policy_ages: Int[Tensor, " seq"] | None = None


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
    # Non-empty rows sum to at least 1 (their max term), so the clamp only keeps empty rows' gradient finite.
    return row_max + total.clamp_min(1.0).log()


def selective_log_softmax_with_support(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"], support: TokenSupport
) -> tuple[Float[Tensor, "batch seq"], Float[Tensor, " entries"]]:
    """Per-token logprobs with support replay, and the logprob of every support entry under the same
    normalization: tokens with a usable sampling mask (see ``support_replay_mask``) get ``logits[index] -
    logsumexp(logits[mask])``, others full-vocab."""
    logits, index = logits.squeeze(0), index.squeeze(0)
    rows = support.row_index().long()
    support_logits = logits[rows, support.ids.long()]
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


def ipo_weight(config: IPOLossConfig, log_p: Tensor, log_q: Tensor) -> tuple[Tensor, Tensor]:
    # |p - q| = e^max * -expm1(min - max), without cancellation near p = q.
    larger, smaller = torch.maximum(log_p, log_q), torch.minimum(log_p, log_q)
    keep = torch.exp(larger) * -torch.expm1(smaller - larger) <= config.eps
    log_ratio = (log_p - log_q).clamp(max=math.log(config.max_importance_ratio))
    return keep, torch.exp(torch.where(keep, log_ratio, 0.0)) * keep


def icepop_weight(config: IcePopLossConfig, log_p: Tensor, log_q: Tensor) -> tuple[Tensor, Tensor]:
    log_ratio = log_p - log_q
    keep = (log_ratio >= math.log(config.ratio_low)) & (log_ratio <= math.log(config.ratio_high))
    return keep, torch.exp(torch.where(keep, log_ratio, 0.0)) * keep


def reinforce_weight(config: ReinforceLossConfig, log_p: Tensor, log_q: Tensor) -> tuple[Tensor, Tensor]:
    keep = torch.ones_like(log_p, dtype=torch.bool)
    return keep, torch.ones_like(log_p)


class RLLoss:
    """Policy gradient with importance weight ``w`` (set by the loss type), score centering under the sampler
    and an optional KL(sampler || trainer), all over one support per token, as sums ``sum_v u_v f(v)`` that
    estimate ``E_q[f]``: the replayed sampling mask (exact, ``u_v = q_v``), K i.i.d. sampler draws
    (``u_v = 1/K``), or a captured top-k head (``u_v = q_v``) plus a tail with trainer probabilities
    proportional to the sampler's."""

    def __init__(self, config: IPOLossConfig | IcePopLossConfig | ReinforceLossConfig):
        self.config = config
        weight_fns = {"ipo": ipo_weight, "icepop": icepop_weight, "reinforce": reinforce_weight}
        self.weight_fn = weight_fns[config.type]

    def weight(self, log_p: Tensor, log_q: Tensor) -> tuple[Tensor, Tensor]:
        return self.weight_fn(self.config, log_p.detach(), log_q.detach())

    def loss(self, inputs: LossInputs) -> LossOutputs:
        config = self.config
        mask = inputs.loss_mask
        logprobs = inputs.trainer_logprobs[mask]
        log_ratio = logprobs - inputs.inference_logprobs[mask]
        advantages = config.adv_tau * inputs.advantages[mask]

        keep, w = self.weight(logprobs, inputs.inference_logprobs[mask])
        # Value w, gradient w * grad log p.
        per_token_loss = -advantages * w * torch.exp(logprobs - logprobs.detach())

        mismatch_kl = _mismatch_kl_from_log_ratio(log_ratio.detach())
        metrics = {
            "masked_mismatch_kl": _safe_mean(mismatch_kl, ~keep),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep),
            "is_masked": (~keep).sum() / max(keep.numel(), 1),
        }
        if inputs.support_rows is not None:
            support_loss, support_metrics = self.support_loss(inputs, advantages)
            per_token_loss = per_token_loss + support_loss
            metrics.update(support_metrics)
        if inputs.loss_weights is not None:
            per_token_loss = per_token_loss * inputs.loss_weights[mask]
        return LossOutputs(loss=per_token_loss.sum(), metrics=metrics)

    def support_loss(self, inputs: LossInputs, advantages: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        config = self.config
        mask = inputs.loss_mask
        rows = inputs.support_rows.long()

        def row_sum(values: Tensor) -> Tensor:
            return values.new_zeros(mask.shape[0]).index_add(0, rows, values)

        log_p = inputs.trainer_support_logprobs
        log_q = inputs.sampler_support_logprobs.detach()
        count = row_sum(torch.ones_like(log_q))
        has_support = count > 0
        replayed = has_support & inputs.support_replayed
        p, q = log_p.exp(), log_q.exp()
        if config.score_centering_estimator == "sampled":
            u = torch.where(replayed[rows], q, 1.0 / count[rows])
            q_tail = torch.zeros_like(count)
        else:
            u = q
            q_tail = torch.where(replayed, 0.0, (1.0 - row_sum(q)).clamp_min(0.0))

        with torch.no_grad():
            _, w = self.weight(log_p, log_q)
            # Tail tokens share one ratio: with p_v = p_tail / q_tail * q_v, sum_tail q_v w_v grad log p_v
            # = -c * sum_head p_v grad log p_v, as sum_v p_v grad log p_v = 0 over the normalized support.
            p_tail = (1.0 - row_sum(p.detach())).clamp_min(1e-6)
            _, w_tail = self.weight(p_tail.log(), q_tail.log())
            c = q_tail * w_tail / p_tail
            mass = u * w - c[rows] * p.detach()
            centered = replayed if config.score_centering == "auto" else has_support
            mass = mass * (centered & (config.score_centering is not False))[rows]
        loss = advantages.detach() * row_sum(mass * log_p)[mask]

        # E_q[p/q] and the TV E_q|p/q - 1| / 2 over the support; the head adds the tail bucket to the TV.
        ratio = (log_p.detach() - log_q).exp()
        head_mass = row_sum(u * ratio)
        p_tail_mass = torch.where(q_tail > 0, (1.0 - head_mass).clamp_min(0.0), 0.0)
        metrics = {
            "score_centering/uncentered": 1.0 - centered[mask].float(),
            "score_centering/tail_mass": q_tail[mask],
            "score_centering/head_mass": head_mass[mask],
            "score_centering/mask_tv": 0.5 * (row_sum((u * (ratio - 1.0)).abs()) + (q_tail - p_tail_mass).abs())[mask],
        }
        if config.kl_coeff == 0:
            return loss, metrics

        # KL(q || p) over the head plus the tail bucket; the tail vanishes on replayed rows.
        p_tail = (1.0 - row_sum(p)).clamp_min(1e-6)
        kl = row_sum(u * (log_q - log_p)) + torch.xlogy(q_tail, q_tail) - q_tail * p_tail.log()
        kl = (kl * has_support)[mask]
        ages = inputs.policy_ages[mask] if inputs.policy_ages is not None else None
        if config.kl_staleness == "inv_sqrt":
            if ages is None or bool((ages < 0).any()):
                raise ValueError("kl_staleness = 'inv_sqrt' requires the sampler policy version of every RL token")
            beta = torch.where(ages >= 1, config.kl_coeff * ages.clamp_min(1).float().rsqrt(), 0.0)
        else:
            beta = torch.full_like(kl, config.kl_coeff)
        metrics |= {"kl": kl.detach(), "kl/beta": beta}
        if ages is not None:
            metrics["kl/age0"] = (ages == 0).float()
        return loss + beta * kl, metrics


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
        case IPOLossConfig() | IcePopLossConfig() | ReinforceLossConfig():
            return RLLoss(loss_config)
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
    policy_ages: list[Tensor] | None = None,
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
        rl_scale: Global sum of rl weights normalizing the rl component
        ce_scale: Global ce-token count normalizing the ce component
        ref_kl_scale: Global ref_kl-token count normalizing the ref_kl component
        support: Token-aligned sampler support of the whole micro batch, or None
        trainer_support_logprobs: Trainer logprobs of every ``support`` entry

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
    policy_ages = policy_ages if policy_ages is not None else [None] * n

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
    for t_logp, i_logp, ref_logp, adv, mask, rl_w, ce_w, ref_kl_w, sample_support, ages in zip(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        sample_supports,
        policy_ages,
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
                policy_ages=ages,
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
