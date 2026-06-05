from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.losses import LossTerm as LossTermConfig
from prime_rl.configs.losses import RLLossConfig, is_primary, to_rl_loss_config
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
    per-token weight. ``scale`` is the term's own (global) token denominator.
    """

    name: str
    core: LossFn
    scale: int
    masks: list[Bool[Tensor, " seq"]]
    weights: list[Float[Tensor, " seq"]]


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

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

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


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def echo_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Weighted masked negative log-likelihood — the echo / SFT-overlay core.

    Reads ``inputs.loss_mask`` as the term's token-selection mask and
    ``inputs.advantages`` as the per-token weight (``alpha``). With weight 1.0 on
    the trainable mask this reduces to plain masked NLL.
    """
    trainer_logprobs = inputs.trainer_logprobs
    mask = inputs.loss_mask
    weight = inputs.advantages

    loss = -(weight * trainer_logprobs)[mask].sum()
    # Only report metrics for splits that actually carry echo tokens. A packed split with no echo
    # tokens would otherwise contribute a spurious echo_nll=0, biasing the mean by packing composition.
    metrics: dict[str, Tensor] = {}
    if mask.any():
        metrics["echo_nll"] = _safe_mean(-trainer_logprobs, mask)
        metrics["echo_token_count"] = mask.sum().float()
    return LossOutputs(loss=loss, metrics=metrics)


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
    - ``"opd"``  → ``opd_loss_fn`` (teacher KL as gradient signal, fixed knobs)
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

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, rl_loss_config)

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
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn],
    loss_scale: int,
    training_mode: str = "rl",
    extra_terms: list[ExtraTerm] | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Loss dispatch is batch-driven: ``training_mode`` selects the loss term(s) via
    ``build_loss_terms`` from the core registry ``loss_fns`` (built by
    ``setup_loss_fns``). Every term is applied and summed before the single
    backward; today there is exactly one term per sample. sft → sft_loss_fn,
    opd → opd_loss_fn, rl → the configured default/custom loss.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fns: Per-mode loss fn dispatch table from setup_loss_fns()
        loss_scale: Scale factor to normalize the primary (training_mode) term
        training_mode: Selects which loss fn to apply
        extra_terms: Additional terms (e.g. echo) summed alongside the primary
            term; each carries its own per-sample mask/weight and scale.

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    primary_terms = build_loss_terms(training_mode, loss_fns)

    total_loss = 0.0
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    # Materialized so extra terms can re-zip the shared per-sample inputs.
    samples = list(zip(trainer_logprobs, inference_logprobs, teacher_logprobs, advantages, loss_mask))

    for t_logp, i_logp, teach_logp, adv, mask in samples:
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )
        for term in primary_terms:
            result = term.core(inputs)
            total_loss = total_loss + result.loss
            _accumulate_metrics(all_metrics, result.metrics)

    # Primary term keeps the single global divide, so the rl-only path stays
    # bit-identical to the pre-term-refactor loss.
    scaled_loss = total_loss / loss_scale

    # Extra terms (e.g. echo) carry their own per-token mask/weight and scale.
    # Every term differentiates the same shared forward -> one backward upstream.
    for term in extra_terms or []:
        term_total = 0.0
        for (t_logp, i_logp, teach_logp, _adv, _mask), term_mask, term_weight in zip(
            samples, term.masks, term.weights, strict=True
        ):
            inputs = LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                teacher_logprobs=teach_logp,
                advantages=term_weight,
                loss_mask=term_mask,
            )
            result = term.core(inputs)
            term_total = term_total + result.loss
            _accumulate_metrics(all_metrics, result.metrics)
        scaled_loss = scaled_loss + term_total / term.scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
