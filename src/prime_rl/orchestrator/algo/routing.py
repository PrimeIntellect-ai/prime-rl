"""Wire-field stamping for per-token component weights and advantage spreading.

The training loss is a sum of three components — ``rl`` (importance-weighted
PG + KL), ``ce`` (masked NLL), and ``ref_kl`` (reverse KL to a reference model
as the PG signal) — each normalized by its own global token count in the
trainer. The advantage strategy decides which component the action tokens feed
and optional per-token advantages; these helpers write the per-token component
weight streams onto the ``TrainingSample`` wire fields at group finalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


def stamp_loss_routing(sample: TrainingSample, action_loss_type: str, observation_weight: float | None) -> None:
    """Stamp the algorithm's loss routing onto one sample's component weight
    streams.

    Action tokens (the trainable completion tokens) feed the algorithm's
    component: ``rl`` is the default (absent streams ship nothing), while
    ``ce``/``ref_kl`` stamp that component's weights over the action tokens
    and zero the rl stream. When the algorithm trains on observations
    (``observation_weight`` is set), env-provided tokens (tagged by
    ``interleave_rollout`` in ``completion_obs_mask``) get that ce weight —
    they stay out of ``completion_mask``, so the ce component is the only one
    that trains them. ``completion_obs_mask`` is orchestrator-internal and
    cleared here so it never ships.
    """
    obs_mask = sample.completion_obs_mask
    sample.completion_obs_mask = None
    train_obs = observation_weight is not None and obs_mask is not None and any(obs_mask)
    if action_loss_type == "rl" and not train_obs:
        return

    prompt_len = len(sample.prompt_ids)
    seq_len = prompt_len + len(sample.completion_ids)

    if action_loss_type != "rl":
        sample.rl_weights = [0.0] * seq_len
        action_weights = [0.0] * seq_len
        for i, trains in enumerate(sample.completion_mask):
            if trains:
                action_weights[prompt_len + i] = 1.0
        if action_loss_type == "ce":
            sample.ce_weights = action_weights
        else:
            assert action_loss_type == "ref_kl"
            sample.ref_kl_weights = action_weights

    if train_obs:
        assert obs_mask is not None and observation_weight is not None
        ce_weights = sample.ce_weights if sample.ce_weights is not None else [0.0] * seq_len
        for i, is_obs in enumerate(obs_mask):
            if is_obs:
                ce_weights[prompt_len + i] = observation_weight
        sample.ce_weights = ce_weights


def spread_token_advantages(rollout: TrainRollout) -> None:
    """Stamp the strategy's per-token advantages onto the rollout's sample,
    padded with 0.0 over prompt positions (never trained).

    Per-token advantages are aligned to one sample's completion tokens, so a
    rollout that split into several training samples is rejected — there is no
    unambiguous way to distribute one list across them.
    """
    assert rollout.token_advantages is not None
    if len(rollout.samples) != 1:
        raise ValueError(
            f"per-token advantages need a rollout with exactly one training sample; "
            f"env '{rollout.env_name}' produced {len(rollout.samples)}."
        )
    sample = rollout.samples[0]
    if len(rollout.token_advantages) != len(sample.completion_ids):
        raise ValueError(
            f"per-token advantages must align with the sample's completion tokens: "
            f"got {len(rollout.token_advantages)}, expected {len(sample.completion_ids)} "
            f"(env '{rollout.env_name}')."
        )
    sample.token_advantages = [0.0] * len(sample.prompt_ids) + list(rollout.token_advantages)
