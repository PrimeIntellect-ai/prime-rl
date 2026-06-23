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
    clobbered — env-provided tokens stay out of ``completion_mask``, so the
    component an algorithm weights them into is the only one that trains
    them.
    """
    sample.obs_spans = None  # orchestrator-internal provenance, never ships
    if action_loss_type == "rl":
        return

    prompt_len = len(sample.prompt_ids)
    seq_len = prompt_len + len(sample.completion_ids)
    sample.rl_weights = [0.0] * seq_len
    action_weights = (
        sample.ce_weights if action_loss_type == "ce" and sample.ce_weights is not None else [0.0] * seq_len
    )
    for i, trains in enumerate(sample.completion_mask):
        if trains:
            action_weights[prompt_len + i] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights


def stamp_advantages(rollout: Rollout) -> None:
    """Stamp the rollout's per-token advantage stream onto its samples' wire
    fields, padded with 0.0 over prompt positions (never trained). The stream
    is aligned to the samples' completion tokens (concatenated in step order)
    and sliced across them. Rollouts with no credit assigned
    (``advantages=None``, e.g. opd/opsd) ship no advantage stream.
    """
    advantages = rollout.advantages
    if advantages is None:
        return
    total = sum(len(sample.completion_ids) for sample in rollout.samples)
    if len(advantages) != total:
        raise ValueError(
            f"advantage stream must align with the rollout's completion tokens: "
            f"got {len(advantages)}, expected {total} (env '{rollout.env_name}')."
        )
    offset = 0
    for sample in rollout.samples:
        num_completion = len(sample.completion_ids)
        sample.advantages = [0.0] * len(sample.prompt_ids) + advantages[offset : offset + num_completion]
        offset += num_completion
