"""Role-conditioned advantage estimation for multi-agent training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from verifiers import rollout_to_member_rollouts
from verifiers.types import MemberRollout

from prime_rl.configs.multi_agent import MultiAgentConfig
from prime_rl.configs.orchestrator import AdvantageConfig, EMAPerMemberAdvantageConfig
from prime_rl.orchestrator.member_generation import is_trainable_member as is_bound_trainable_member
from prime_rl.orchestrator.types import RAEStats

MemberTrainability = Callable[[Mapping, str], bool]

RAEKey = tuple[str, int | str, str]


@dataclass
class RAEState:
    """Persistent EMA baselines keyed by (env_name, example_id, member_id)."""

    baselines: dict[RAEKey, float] = field(default_factory=dict)
    momentum: float = 0.9

    def update(self, key: RAEKey, reward: float) -> None:
        prev = self.baselines.get(key, 0.0)
        self.baselines[key] = self.momentum * prev + (1 - self.momentum) * reward


def fan_out_for_multi_agent(
    rollouts: list[Mapping],
    *,
    is_trainable_member: MemberTrainability | None = None,
) -> tuple[list[MemberRollout], list[list[int]]]:
    """Project episode rollouts to per-member training units."""
    training_units: list[MemberRollout] = []
    rollout_to_unit_idxs: list[list[int]] = []
    for rollout in rollouts:
        members = rollout_to_member_rollouts(rollout)
        if is_trainable_member is not None:
            members = [member for member in members if is_trainable_member(rollout, member["member_id"])]
        env_name = rollout.get("env_name")
        if env_name is not None:
            if not isinstance(env_name, str):
                raise TypeError(f"rollout env_name must be a string for RAE identity, got {type(env_name).__name__}")
            for member in members:
                member["env_name"] = env_name
        rollout_to_unit_idxs.append(list(range(len(training_units), len(training_units) + len(members))))
        training_units.extend(members)
    return training_units, rollout_to_unit_idxs


def fan_out_trainable_for_multi_agent(
    rollouts: list[Mapping],
    config: MultiAgentConfig,
) -> tuple[list[MemberRollout], list[list[int]]]:
    """Project episode rollouts to the member rows that can enter training."""
    return fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda rollout, member_id: is_bound_trainable_member(config, rollout, member_id),
    )


def _rae_key(member_rollout: Mapping) -> RAEKey:
    env_name = member_rollout.get("env_name")
    if env_name is None:
        env_name = member_rollout["task"]
    if not isinstance(env_name, str):
        raise TypeError(
            "RAE baseline identity requires a string env_name; "
            f"got {type(env_name).__name__} from member rollout identity fields"
        )
    return (env_name, member_rollout["example_id"], member_rollout["member_id"])


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> tuple[list[float], RAEStats]:
    """Compute per-member advantages and update EMA baselines.

    SPIRAL Alg.1 updates the baseline first, then subtracts that updated
    baseline from the same rollout's reward.

    Also returns an ``RAEStats`` observability snapshot for this call,
    derived from the estimator's inputs/outputs only (no estimator
    internals beyond reading baseline values).
    """
    advantages: list[float] = []
    stats = RAEStats()
    for member_rollout in member_rollouts:
        reward = member_rollout["reward"]
        key = _rae_key(member_rollout)
        prev_baseline = state.baselines.get(key)
        state.update(key, reward)
        baseline = state.baselines[key]
        advantages.append(reward - baseline)

        member_id = member_rollout["member_id"]
        stats.updates += 1
        if prev_baseline is None:
            stats.cold_updates += 1
        stats.baseline_abs_delta_sum += abs(baseline - (prev_baseline if prev_baseline is not None else 0.0))
        stats.baseline_sum_by_member[member_id] = stats.baseline_sum_by_member.get(member_id, 0.0) + baseline
        stats.updates_by_member[member_id] = stats.updates_by_member.get(member_id, 0) + 1
    stats.baseline_keys_total = len(state.baselines)
    return advantages, stats


def validate_advantage_mode(env_name: str, *, is_multi_agent: bool, advantage: AdvantageConfig | None) -> None:
    """Enforce the env ⟺ advantage pairing at startup, once ``is_multi_agent``
    is known (config-time validation cannot see the loaded env): multi-agent
    envs need RAE (``ema_per_member``); a single-agent env resolving to it
    would silently degrade to raw-reward REINFORCE (no baseline)."""
    is_ema = isinstance(advantage, EMAPerMemberAdvantageConfig)
    if is_multi_agent and not is_ema:
        resolved = "None" if advantage is None else advantage.type
        raise ValueError(
            f"Train env {env_name!r} is multi-agent but resolves advantage={resolved!r}. "
            "Multi-agent training requires advantage.type='ema_per_member' — set it in "
            "[orchestrator.advantage] or as this env's per-env advantage override."
        )
    if is_ema and not is_multi_agent:
        raise ValueError(
            f"Train env {env_name!r} resolves advantage.type='ema_per_member' but is not a multi-agent env. "
            "Its groups would train with no baseline (raw-reward REINFORCE). Give this env a per-env "
            'advantage override in its [[orchestrator.train.env]] block, e.g. advantage = { type = "default" }.'
        )
