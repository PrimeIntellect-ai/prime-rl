"""Role-conditioned Advantage Estimation (RAE / SPIRAL).

Separate from the GRPO advantage path (advantage.py) — different computation
model (per-sample with metadata vs. [P,N] tensor), different state lifecycle.
The orchestrator multi-agent path calls compute_rae_advantages instead of
compute_advantages. Both coexist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from verifiers import rollout_to_member_rollouts
from verifiers.types import MemberRollout

RAEKey = tuple[str, int | str, str]
"""(task, example_id, member_id). ``task`` is the env name (MemberRollout['task']),
which partitions the baseline store across environments — otherwise two envs
with overlapping example_ids would contaminate each other's baselines.
``example_id`` is int|str to match MemberRollout (HuggingFace UID-style ids
like \"mmlu_0001\" propagate verbatim end-to-end). ``member_id`` is the single
participant label after the verifiers α-cut (role_id was deleted as a
redundant duplicate label)."""


@dataclass
class RAEState:
    """Persistent EMA baselines keyed by (task, example_id, member_id).

    Cold start: missing keys default to 0.0 baseline, so the first
    advantage for a new (task, example, member) tuple equals the raw reward.
    """

    baselines: dict[RAEKey, float] = field(default_factory=dict)
    momentum: float = 0.9

    def update(self, key: RAEKey, reward: float) -> None:
        prev = self.baselines.get(key, 0.0)
        self.baselines[key] = self.momentum * prev + (1 - self.momentum) * reward


def fan_out_for_multi_agent(
    rollouts: list[Mapping],
    *,
    drop_judge: bool = True,
    filter_by_learner_seat: bool = False,
) -> tuple[list[MemberRollout], list[list[int]]]:
    """Project episode-level rollouts to per-member training units.

    Returns ``(training_units, rollout_to_unit_idxs)``. The latter maps each
    input rollout index to the indices of its member units in
    ``training_units`` — the orchestrator uses it to fold per-unit token
    counts back to per-rollout metric rows (results_df is per-episode).

    ``drop_judge`` filters out ``member_id == "judge"`` units. By the
    ``zero_sum_reward`` construction the judge gets reward 0 for every
    episode; including it in training only burns gradient compute against
    a zero-signal advantage. Pass ``False`` only for diagnostic runs that
    want judge transcripts in the training batch (e.g. SFT-on-judge).

    ``filter_by_learner_seat`` keeps only the member matching
    ``rollout.info['learner_seat']``. Used by external-opponent training
    (the opposite seat and judge are frozen external endpoints whose
    trajectories are dead weight on the training path). Raises when the
    key is missing -- enabling the filter on a self-play env is a config
    mismatch, not a silent no-op.
    """
    training_units: list[MemberRollout] = []
    rollout_to_unit_idxs: list[list[int]] = []
    for rollout in rollouts:
        members = rollout_to_member_rollouts(rollout)
        if drop_judge:
            members = [m for m in members if m["member_id"] != "judge"]
        if filter_by_learner_seat:
            info = rollout.get("info") or {}
            seat = info.get("learner_seat")
            if seat is None:
                raise ValueError(
                    "fan_out_for_multi_agent: filter_by_learner_seat=True but "
                    "rollout.info['learner_seat'] is missing "
                    f"(example_id={rollout.get('example_id')!r}). The env-pack "
                    "must stamp info.learner_seat per row (e.g. gpqa_debate "
                    "with opponent_model set). Disable the filter for "
                    "self-play envs."
                )
            members = [m for m in members if m["member_id"] == seat]
        rollout_to_unit_idxs.append(list(range(len(training_units), len(training_units) + len(members))))
        training_units.extend(members)
    return training_units, rollout_to_unit_idxs


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> list[float]:
    """Compute per-member advantages and update EMA baselines per SPIRAL Alg.1.

    For each rollout in order:
      b[(task, example_id, member_id)] ← α·b + (1-α)·R       (Alg.1, line 20)
      A(τ) = R(τ) - b[(task, example_id, member_id)]         (Alg.1, line 21)

    Update-then-subtract, per-trajectory. Within-batch order matters when
    multiple rollouts share a key — the recursion is the point. Each
    trajectory's advantage uses the baseline that has just absorbed its
    own reward; downstream trajectories then see a baseline weighted toward
    their predecessors' rewards.
    """
    advantages: list[float] = []
    for mr in member_rollouts:
        reward = mr["reward"]
        key: RAEKey = (mr["task"], mr["example_id"], mr["member_id"])
        state.update(key, reward)
        advantages.append(reward - state.baselines[key])
    return advantages
