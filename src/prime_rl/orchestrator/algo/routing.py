"""Wire-field stamping for the per-token streams.

The training loss is a sum of three components — ``rl`` (importance-weighted
PG + KL), ``ce`` (masked NLL), and ``ref_kl`` (reverse KL to a reference model
as the PG signal) — each normalized by its own global token count in the
trainer. The advantage strategy decides which component the action tokens feed
and the per-token advantages the rl component consumes; these helpers write
the component weight streams and the advantage stream onto the
``TrainingSample`` wire fields at group finalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def stamp_loss_routing(sample: TrainingSample, action_loss_type: ActionLossType) -> None:
    """Stamp the algorithm's loss routing onto one sample's component weight
    streams: action tokens (the trainable completion tokens, per the loss
    mask) feed the algorithm's declared component.

    ``rl`` is the default and ships nothing (absent streams mean rl weight
    1.0 on the loss mask — the hot path); ``ce``/``ref_kl`` weight the action
    tokens into that component's stream and zero the rl stream. Streams an
    algorithm wrote directly (echo's observation ce weights) are merged, not
    clobbered — env-provided tokens stay out of the loss ``mask``, so the
    component an algorithm weights them into is the only one that trains
    them.
    """
    if action_loss_type == "rl":
        return

    seq_len = len(sample.token_ids)
    sample.rl_weights = [0.0] * seq_len
    action_weights = (
        sample.ce_weights if action_loss_type == "ce" and sample.ce_weights is not None else [0.0] * seq_len
    )
    for i, trains in enumerate(sample.mask):
        if trains:
            action_weights[i] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights


def stamp_advantages(rollout: Rollout) -> None:
    """Stamp the rollout's per-token advantage stream onto its samples' wire
    fields. The stream is full-length-N — aligned to the samples' ``token_ids``
    concatenated in order, 0.0 on non-trainable positions — and sliced across
    them. Rollouts with no credit assigned (``advantages=None``, e.g. opd/opsd)
    ship no advantage stream.
    """
    advantages = rollout.advantages
    if advantages is None:
        return
    total = sum(len(sample.token_ids) for sample in rollout.samples)
    if len(advantages) != total:
        raise ValueError(
            f"advantage stream must align with the rollout's tokens: "
            f"got {len(advantages)}, expected {total} (env '{rollout.env_name}')."
        )
    offset = 0
    for sample in rollout.samples:
        num_tokens = len(sample.token_ids)
        sample.advantages = list(advantages[offset : offset + num_tokens])
        offset += num_tokens
