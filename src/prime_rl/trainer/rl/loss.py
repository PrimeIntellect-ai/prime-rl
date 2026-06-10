import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.losses import HookConfig, ReduceConfig, RLLossConfig, is_primary, to_rl_loss_config
from prime_rl.configs.losses import LossTerm as LossTermConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    reference_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample.

    ``loss`` is the per-sample scalar (the masked, summed loss) — what the no-hook path consumes
    directly, so it stays bit-identical to the pre-hook behaviour. ``per_token_loss`` is the same loss
    *before* summation (full sequence length, ``0`` on masked tokens); it's what the hook chain
    transforms when a term has hooks. Built-in cores populate both; a custom core only needs
    ``per_token_loss`` if hooks are configured on its term."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
    per_token_loss: Float[Tensor, " seq"] | None = None


@dataclass
class ReduceInputs:
    """Inputs to a per-term reduce: the per-sample (already λ-weighted) summed losses, their
    per-sample eligibility masks, and the global (all-reduced) eligible-token count.

    ``mean_reduce`` (the default) uses ``global_scale`` for a true global per-token mean. A custom
    reduce may instead use the per-sample data (e.g. normalize each sequence on its own), but is then
    responsible for cross-rank normalization — ``global_scale`` is the only globally-reduced input.
    """

    per_sample_losses: list[Float[Tensor, ""]]
    per_sample_eligible: list[Bool[Tensor, " seq"]]
    global_scale: int


Reduce = Callable[["ReduceInputs"], Tensor]
"""Reduces a term's per-sample losses to one scalar — the term's normalization step."""


def mean_reduce(inputs: ReduceInputs) -> Tensor:
    """Global per-token mean: sum the per-sample losses and divide by the global eligible count.

    Bit-identical to the historical ``total_loss / loss_scale`` normalization.
    """
    return sum(inputs.per_sample_losses) / inputs.global_scale


def setup_reduce(config: ReduceConfig) -> Reduce:
    """Resolve a loss term's ``reduce`` axis config to a normalization callable (the trainer runs it
    per term in ``compute_loss``). ``mean`` is the global per-token mean; ``custom`` is an import path."""
    if config.type == "mean":
        return mean_reduce
    fn = import_object(config.import_path)
    return functools.partial(fn, **config.kwargs)


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


Hook = Callable[[Tensor, LossInputs], Tensor]
"""A per-term, trainer-side post-core transform: ``hook(per_token_loss, inputs) -> per_token_loss``.

Chainable, runs between the core and the term's reduce, and sees all trainer-side per-token data via
``inputs`` (the live ``trainer_logprobs`` / ``inference_logprobs`` / ``advantages`` / ``loss_mask``).
No scalar return — reduction is the separate reduce step, so masking/gating hooks compose. For
trainer-side signals that can't be precomputed orchestrator-side (current-policy prob/entropy gating,
smoothing, penalties); intrinsic objective math (DPPO clip + KL) stays inside the core."""


def min_prob_filter(per_token_loss: Tensor, inputs: LossInputs, *, min_logprob: float) -> Tensor:
    """Built-in hook: zero the per-token loss where the current-policy logprob is below ``min_logprob``
    (a trainer-side filter — it reads the live forward, so it can't be precomputed orchestrator-side)."""
    return per_token_loss * (inputs.trainer_logprobs >= min_logprob)


def setup_hooks(configs: list[HookConfig]) -> list[Hook]:
    """Resolve a loss term's ``hooks`` to a chain of trainer-side per-token transforms, applied in
    order: a built-in preset (``min_prob_filter``) or a ``custom`` import path."""
    hooks: list[Hook] = []
    for config in configs:
        if config.type == "min_prob_filter":
            hooks.append(functools.partial(min_prob_filter, min_logprob=config.min_logprob))
        else:
            hooks.append(functools.partial(import_object(config.import_path), **config.kwargs))
    return hooks


@dataclass
class LossTerm:
    """A single loss term: a named core loss fn applied to a packed sample.

    ``compute_loss`` applies every term in its list and sums the results before a
    single backward. Today there is exactly one term per sample (selected by
    ``training_mode``); this is the seam where additional terms are added.
    """

    name: str
    core: LossFn


@dataclass
class ExtraTerm:
    """An additional loss term applied alongside the primary (training_mode) term.

    ``masks``/``weights`` are per packed sample (parallel to ``compute_loss``'s
    ``trainer_logprobs``). The core sees a ``LossInputs`` whose ``loss_mask`` is
    this term's token-selection mask and whose ``advantages`` carry this term's
    per-token weight. ``scale`` is the term's own (global) token denominator;
    ``lambda_weight`` scales its contribution (pre-reduce) and ``reduce`` is its
    normalization step.
    """

    name: str
    core: LossFn
    scale: int
    masks: list[Bool[Tensor, " seq"]]
    weights: list[Float[Tensor, " seq"]]
    lambda_weight: float = 1.0
    reduce: Reduce = mean_reduce
    hooks: list[Hook] = field(default_factory=list)


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


def _accumulate_metrics(all_metrics: dict[str, list[Tensor]], metrics: dict[str, Tensor]) -> None:
    for k, v in metrics.items():
        all_metrics.setdefault(k, []).append(v)


def _term_sample_loss(result: LossOutputs, hooks: list[Hook], inputs: LossInputs, name: str) -> Tensor:
    """A term's per-sample scalar loss. With no hooks this is exactly ``result.loss`` (the core's
    masked sum — bit-identical to the pre-hook path). With hooks, the core's ``per_token_loss`` is
    passed through the chain and summed."""
    if not hooks:
        return result.loss
    per_token_loss = result.per_token_loss
    if per_token_loss is None:
        raise ValueError(
            f"loss term {name!r} has hooks configured but its core returned no per_token_loss; "
            "a hookable core must populate LossOutputs.per_token_loss."
        )
    for hook in hooks:
        per_token_loss = hook(per_token_loss, inputs)
    return per_token_loss.sum()


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def default_loss_fn(inputs: LossInputs, loss_config: RLLossConfig) -> LossOutputs:
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

    # advantages are already the resolved weight (GRPO advantage × the advantage-weight's tau,
    # applied orchestrator-side), so the core consumes them directly.
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
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

    return LossOutputs(loss=loss, per_token_loss=per_token_loss, metrics=metrics)


def pg_loss_fn(
    inputs: LossInputs,
    *,
    use_importance_ratio: bool = True,
    clip: tuple[float, float] | None = (0.2, 0.2),
    kl_weight: float = 1e-3,
) -> LossOutputs:
    """Parameterizable policy-gradient core — the single core that rl / echo / sft collapse into.

    - ``use_importance_ratio``: weight the per-token gradient by the importance ratio
      ``exp(trainer_lp - inference_lp)`` (on-policy RL) when True, or by the raw ``trainer_lp``
      (weighted masked NLL) when False.
    - ``clip=(low, high)``: apply the DPPO trust-region mask (advantage-sign-conditioned); ``None``
      disables it.
    - ``kl_weight``: scale of the squared-KL regularizer (``0`` disables it).

    At ``(True, (dppo_mask_low, dppo_mask_high), kl_tau)`` this is bit-identical to ``default_loss_fn``;
    at ``(False, None, 0.0)`` it is ``echo_loss_fn`` (and ``sft_loss_fn`` at unit weight). The advantage
    is the resolved per-token weight (computed orchestrator-side) and is consumed directly.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )
    # On-policy RL weights the gradient by the importance ratio; NLL (echo/sft) by the raw logprob.
    policy_term = importance_ratio if use_importance_ratio else trainer_logprobs

    if clip is not None:
        dppo_mask_low, dppo_mask_high = clip
        probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
        dppo_invalid_mask_high = probs_diff > dppo_mask_high
        dppo_invalid_mask_low = probs_diff < -dppo_mask_low
        positive_advantages = advantages > 0
        negative_advantages = advantages < 0
        is_masked = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)
        is_masked_high = positive_advantages & dppo_invalid_mask_high
        is_masked_low = negative_advantages & dppo_invalid_mask_low
        keep_mask = loss_mask & ~is_masked
        drop_mask = loss_mask & is_masked
        metrics = {
            "masked_mismatch_kl": _safe_mean(mismatch_kl, drop_mask),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            "is_masked": _safe_mean(is_masked, loss_mask),
            "is_masked_low": _safe_mean(is_masked_low, loss_mask),
            "is_masked_high": _safe_mean(is_masked_high, loss_mask),
            "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
            "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        }
    else:
        keep_mask = loss_mask
        metrics = {}
        if loss_mask.any():
            metrics["nll"] = _safe_mean(-trainer_logprobs, loss_mask)
            metrics["token_count"] = loss_mask.sum().float()

    pg_loss = keep_mask * advantages * policy_term
    loss = -pg_loss
    if kl_weight != 0.0:
        loss = loss + kl_weight * (loss_mask * log_importance_ratio**2)
    return LossOutputs(loss=loss.sum(), per_token_loss=loss, metrics=metrics)


def opd_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    On-policy distillation loss: the default DPPO+KL math with the tau knobs
    hardcoded to drop the reward signal and use the reference KL as the
    per-token policy-gradient signal.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    reference_logprobs = inputs.reference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    if reference_logprobs is None:
        raise ValueError("opd_loss_fn requires reference_logprobs - configure a reference for opd mode.")

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

    reference_kl = reference_logprobs - trainer_logprobs
    advantages = 0.0 * advantages + 1.0 * reference_kl.detach()

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + 1e-3 * kl_loss
    loss = per_token_loss.sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        "reference_kl": _safe_mean(reference_kl, loss_mask),
    }

    return LossOutputs(loss=loss, per_token_loss=per_token_loss, metrics=metrics)


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    per_token_loss = -(loss_mask * trainer_logprobs)
    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, per_token_loss=per_token_loss, metrics=metrics)


def echo_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Weighted masked negative log-likelihood — the echo / SFT-overlay core.

    Reads ``inputs.loss_mask`` as the term's token-selection mask and
    ``inputs.advantages`` as the per-token weight (``alpha``). With weight 1.0 on
    the trainable mask this reduces to plain masked NLL.
    """
    trainer_logprobs = inputs.trainer_logprobs
    mask = inputs.loss_mask
    weight = inputs.advantages

    per_token_loss = -(mask * weight * trainer_logprobs)
    loss = -(weight * trainer_logprobs)[mask].sum()
    # Only report metrics for splits that actually carry echo tokens. A packed split with no echo
    # tokens would otherwise contribute a spurious echo_nll=0, biasing the mean by packing composition.
    metrics: dict[str, Tensor] = {}
    if mask.any():
        metrics["echo_nll"] = _safe_mean(-trainer_logprobs, mask)
        metrics["echo_token_count"] = mask.sum().float()
    return LossOutputs(loss=loss, per_token_loss=per_token_loss, metrics=metrics)


def _make_custom_core(import_path: str, kwargs: dict) -> LossFn:
    """Wrap an imported ``core(inputs, **kwargs) -> LossOutputs`` into a no-kwargs ``LossFn``."""
    fn = import_object(import_path)

    def core(inputs: LossInputs) -> LossOutputs:
        return fn(inputs, **kwargs)

    return core


def setup_loss_fns(losses: list[LossTermConfig]) -> dict[str, LossFn]:
    """Build the per-training-mode core registry from the loss-term list.

    The trainer routes per batch from ``TrainingSample.training_mode``:

    - ``"sft"``  → ``sft_loss_fn`` (masked NLL)
    - ``"opd"``  → ``opd_loss_fn`` (reference KL as gradient signal, fixed knobs)
    - ``"rl"``   → ``default_loss_fn`` configured by the primary ``dppo_kl`` term, or a
      ``custom`` core's imported function.
    - ``"echo"`` → ``echo_loss_fn`` (weighted CE), applied by additive echo terms.

    Only the primary (dppo_kl/custom core) term affects the rl path; sft/opd/echo cores are fixed.
    """
    primary = next((term for term in losses if is_primary(term)), None)
    if primary is None:
        # No primary term: don't fabricate a default. An rl-mode batch erroring here is
        # the trainer-side complement to the orchestrator's training_mode/losses validation.
        def rl_fn(inputs: LossInputs) -> LossOutputs:
            raise ValueError(
                "rl-mode batch received but `losses` has no primary (dppo_kl/custom) term. Add one, or set "
                "the orchestrator's training_mode to match the configured terms."
            )
    elif primary.loss.type == "custom":
        rl_fn = _make_custom_core(primary.loss.import_path, primary.loss.kwargs)
    else:
        rl_loss_config = to_rl_loss_config(primary)
        # The rl core is the parameterizable pg core at the rl preset — bit-identical to the historical
        # default_loss_fn (guarded by test_pg_core_matches_default_loss_fn_at_rl_preset).
        rl_fn = functools.partial(
            pg_loss_fn,
            use_importance_ratio=True,
            clip=(rl_loss_config.dppo_mask_low, rl_loss_config.dppo_mask_high),
            kl_weight=rl_loss_config.kl_tau,
        )

    # sft/opd/rl are the training_mode-dispatched primary cores; each remaining (overlay) term
    # contributes an additive core keyed by its name (ce → weighted masked NLL; custom → imported fn).
    cores: dict[str, LossFn] = {"sft": sft_loss_fn, "opd": opd_loss_fn, "rl": rl_fn}
    for term in losses:
        if is_primary(term):
            continue
        cores[term.name] = (
            echo_loss_fn if term.loss.type == "ce" else _make_custom_core(term.loss.import_path, term.loss.kwargs)
        )
    return cores


def build_loss_terms(training_mode: str, cores: dict[str, LossFn]) -> list[LossTerm]:
    """Select the loss term(s) for a packed sample.

    Currently one term per sample, keyed by ``training_mode`` into the core
    registry built by ``setup_loss_fns``.
    """
    try:
        core = cores[training_mode]
    except KeyError:
        raise ValueError(
            f"No loss fn available for training_mode={training_mode!r} "
            f"(available: {sorted(cores)}). Check the sample's training_mode."
        )
    return [LossTerm(name=training_mode, core=core)]


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    reference_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn],
    loss_scale: int,
    training_mode: str = "rl",
    extra_terms: list[ExtraTerm] | None = None,
    reduce: Reduce = mean_reduce,
    primary_lambda: float = 1.0,
    primary_hooks: list[Hook] | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Loss dispatch is batch-driven: ``training_mode`` selects the loss term(s) via ``build_loss_terms``
    from the core registry ``loss_fns`` (built by ``setup_loss_fns``). Each term produces one summed
    loss per sample, is scaled by its weight (``primary_lambda`` / ``ExtraTerm.lambda_weight``,
    pre-reduce), and is reduced to a scalar by its ``reduce`` (default ``mean_reduce`` = divide by the
    global token count). Terms are summed before the single backward; today there is exactly one
    primary term per sample.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        reference_logprobs: Reference log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fns: Per-mode loss fn dispatch table from setup_loss_fns()
        loss_scale: Global eligible-token count for the primary term's reduce
        training_mode: Selects which loss fn to apply
        extra_terms: Additional terms (e.g. echo) summed alongside the primary term; each carries its
            own per-sample mask/weight, λ, scale, and reduce.
        reduce: The primary term's reduce (normalization) step. Default: global per-token mean.
        primary_lambda: Scalar weight on the primary term, applied pre-reduce. Default 1.0.

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    primary_terms = build_loss_terms(training_mode, loss_fns)
    all_metrics: dict[str, list[Tensor]] = {}

    if reference_logprobs is None:
        reference_logprobs = [None] * len(trainer_logprobs)

    # Materialized so extra terms can re-zip the shared per-sample inputs.
    samples = list(zip(trainer_logprobs, inference_logprobs, reference_logprobs, advantages, loss_mask))

    # Primary (training_mode) term: one summed loss per sample (x lambda), reduced to a scalar. The
    # default mean_reduce divides by loss_scale, so the rl-only path stays bit-identical to the loss
    # before the reduce/lambda seam.
    primary_losses: list[Tensor] = []
    primary_eligible: list[Tensor] = []
    for t_logp, i_logp, ref_logp, adv, mask in samples:
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            reference_logprobs=ref_logp,
            advantages=adv,
            loss_mask=mask,
        )
        sample_loss = 0.0
        for term in primary_terms:
            result = term.core(inputs)
            sample_loss = sample_loss + _term_sample_loss(result, primary_hooks or [], inputs, term.name)
            _accumulate_metrics(all_metrics, result.metrics)
        primary_losses.append(primary_lambda * sample_loss)
        primary_eligible.append(mask)
    scaled_loss = reduce(ReduceInputs(primary_losses, primary_eligible, loss_scale))

    # Extra terms (e.g. echo) carry their own per-token mask/weight, lambda, scale, and reduce.
    # Every term differentiates the same shared forward -> one backward upstream.
    for term in extra_terms or []:
        term_losses: list[Tensor] = []
        term_eligible: list[Tensor] = []
        for (t_logp, i_logp, ref_logp, _adv, _mask), term_mask, term_weight in zip(
            samples, term.masks, term.weights, strict=True
        ):
            inputs = LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                reference_logprobs=ref_logp,
                advantages=term_weight,
                loss_mask=term_mask,
            )
            result = term.core(inputs)
            term_losses.append(term.lambda_weight * _term_sample_loss(result, term.hooks, inputs, term.name))
            term_eligible.append(term_mask)
            # Namespace overlay metrics by term so multiple overlays (or custom cores) can't collide
            # with each other or with the primary's metrics.
            _accumulate_metrics(all_metrics, {f"{term.name}/{k}": v for k, v in result.metrics.items()})
        scaled_loss = scaled_loss + term.reduce(ReduceInputs(term_losses, term_eligible, term.scale))

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
