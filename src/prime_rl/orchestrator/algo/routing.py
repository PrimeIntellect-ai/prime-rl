"""Wire-field stamping for per-token loss routing and advantage spreading.

The advantage strategy decides each sample's action loss type and optional
per-token advantages; these helpers write them onto the ``TrainingSample``
wire fields at group finalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import LossRoutingConfig
from prime_rl.transport import TrainingSample
from prime_rl.transport.types import LossType

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout

ACTION_LOSS_TYPES = {"rl": LossType.RL, "ce": LossType.CE, "ref_kl": LossType.REF_KL}


def stamp_loss_routing(sample: TrainingSample, action_loss_type: LossType, loss: LossRoutingConfig) -> None:
    """Stamp the env's loss routing onto one sample's wire fields.

    Action tokens (the trainable completion tokens) get the advantage
    strategy's loss type. When the algorithm trains on observations,
    env-provided tokens (tagged by ``interleave_rollout`` in
    ``completion_obs_mask``) flip from masked-out to trainable on the CE loss type
    with ``observation_weight``. ``completion_obs_mask`` is
    orchestrator-internal and cleared here so it never ships.
    """
    sample.loss_type = action_loss_type
    obs_mask = sample.completion_obs_mask
    sample.completion_obs_mask = None
    if loss.observation == "none" or obs_mask is None or not any(obs_mask):
        return

    prompt_len = len(sample.prompt_ids)
    seq_len = prompt_len + len(sample.completion_ids)
    type_ids = [action_loss_type] * seq_len
    weights = [1.0] * seq_len
    completion_mask = list(sample.completion_mask)
    for i, is_obs in enumerate(obs_mask):
        if is_obs:
            type_ids[prompt_len + i] = LossType.CE
            weights[prompt_len + i] = loss.observation_weight
            completion_mask[i] = True
    sample.completion_mask = completion_mask
    sample.token_loss_types = type_ids
    sample.token_loss_weights = weights


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
