"""Wire-field stamping for the per-token streams.

The training loss is a sum of three components — ``rl`` (importance-weighted
PG + KL), ``ce`` (masked NLL), and ``ref_kl`` (reverse KL to a reference model
as the PG signal) — each normalized by its own global token count in the
trainer. The algorithm decides which component the action tokens feed
and the per-token advantages the rl component consumes; these helpers write
the component weight streams and the advantage stream onto the
``TrainingSample`` wire fields at group finalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


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


def stamp_advantages(rollout: TrainRollout) -> None:
    """Copy each trainable branch's per-token credit onto the sample built from it, zeroed where
    the sample does not train. The branch spreads its nodes' values across its own tokens, so the
    two align by construction, but a node shared with an earlier branch is credited there and is
    only context here. A rollout that was never scored (opd/opsd) ships no advantage stream."""
    for sample, (branch, _) in zip(rollout.samples, iter_trainable_branches(rollout), strict=True):
        advantages = branch.advantages
        if advantages is None:
            continue
        sample.advantages = [a if trains else 0.0 for a, trains in zip(advantages, sample.mask, strict=True)]
