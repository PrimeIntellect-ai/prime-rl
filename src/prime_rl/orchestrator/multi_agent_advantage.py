"""Role-conditioned Advantage Estimation (RAE / SPIRAL).

Separate from the GRPO advantage path (advantage.py) — different computation
model (per-sample with metadata vs. [P,N] tensor), different state lifecycle.
The orchestrator multi-agent path calls compute_rae_advantages instead of
compute_advantages. Both coexist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from verifiers import rollout_to_member_rollouts
from verifiers.types import MemberRollout

MemberTrainability = Callable[[Mapping, str], bool]

RAEKey = tuple[str, int | str, str]
"""(env_name, example_id, member_id). ``env_name`` partitions the baseline
store across environments — otherwise two envs with overlapping example_ids
would contaminate each other's baselines.
``example_id`` is int|str to match MemberRollout (HuggingFace UID-style ids
like \"mmlu_0001\" propagate verbatim end-to-end). ``member_id`` is the single
participant label after the verifiers α-cut (role_id was deleted as a
redundant duplicate label)."""


@dataclass
class RAEState:
    """Persistent EMA baselines keyed by (env_name, example_id, member_id).

    Cold start: missing keys default to 0.0 baseline, so the first
    advantage for a new (env, example, member) tuple equals the raw reward.
    """

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
    """Project episode-level rollouts to per-member training units.

    Returns ``(training_units, rollout_to_unit_idxs)``. The latter maps each
    input rollout index to the indices of its member units in
    ``training_units`` — the orchestrator uses it to fold per-unit token
    counts back to per-rollout metric rows (results_df is per-episode).

    ``is_trainable_member`` is a runtime-owned predicate keyed by
    ``(rollout, member_id)``. Verifiers provides protocol member ids; Prime-RL
    decides which of those members map to trainable policy parameters.
    """
    training_units: list[MemberRollout] = []
    rollout_to_unit_idxs: list[list[int]] = []
    for rollout in rollouts:
        members = rollout_to_member_rollouts(rollout)
        if is_trainable_member is not None:
            members = [m for m in members if is_trainable_member(rollout, m["member_id"])]
        env_name = rollout.get("env_name")
        if env_name is not None:
            if not isinstance(env_name, str):
                raise TypeError(f"rollout env_name must be a string for RAE identity, got {type(env_name).__name__}")
            for member in members:
                member["env_name"] = env_name
        rollout_to_unit_idxs.append(list(range(len(training_units), len(training_units) + len(members))))
        training_units.extend(members)
    return training_units, rollout_to_unit_idxs


def _rae_key(mr: Mapping) -> RAEKey:
    env_name = mr.get("env_name")
    if env_name is None:
        env_name = mr["task"]
    if not isinstance(env_name, str):
        raise TypeError(
            "RAE baseline identity requires a string env_name; "
            f"got {type(env_name).__name__} from member rollout identity fields"
        )
    return (env_name, mr["example_id"], mr["member_id"])


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> list[float]:
    """Compute per-member advantages and update EMA baselines per SPIRAL Alg.1.

    For each rollout in order:
      b[(env_name, example_id, member_id)] ← α·b + (1-α)·R   (Alg.1, line 20)
      A(τ) = R(τ) - b[(env_name, example_id, member_id)]     (Alg.1, line 21)

    Update-then-subtract, per-trajectory. Within-batch order matters when
    multiple rollouts share a key — the recursion is the point. Each
    trajectory's advantage uses the baseline that has just absorbed its
    own reward; downstream trajectories then see a baseline weighted toward
    their predecessors' rewards.
    """
    advantages: list[float] = []
    for mr in member_rollouts:
        reward = mr["reward"]
        key = _rae_key(mr)
        state.update(key, reward)
        advantages.append(reward - state.baselines[key])
    return advantages
