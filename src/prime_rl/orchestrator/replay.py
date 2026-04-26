from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.utils.utils import mean


@dataclass
class ReplayGroup:
    rollouts: list[vf.RolloutOutput]
    env_name: str
    policy_step: int
    insert_step: int
    num_rollouts: int
    num_tokens: int
    num_trainable_rollouts: int
    abs_advantage_mean: float
    abs_advantage_max: float

    @classmethod
    def from_rollouts(cls, rollouts: list[vf.RolloutOutput], insert_step: int) -> "ReplayGroup":
        if not rollouts:
            raise ValueError("ReplayGroup requires at least one rollout")

        env_name = rollouts[0]["env_name"]
        policy_step = rollouts[0]["policy_step"]
        if any(rollout["env_name"] != env_name for rollout in rollouts):
            raise ValueError("ReplayGroup rollouts must all belong to the same environment")
        if any(rollout["policy_step"] != policy_step for rollout in rollouts):
            raise ValueError("ReplayGroup rollouts must all share the same policy step")

        abs_advantages = [abs(float(rollout.get("advantage") or 0.0)) for rollout in rollouts]
        return cls(
            rollouts=list(rollouts),
            env_name=env_name,
            policy_step=policy_step,
            insert_step=insert_step,
            num_rollouts=len(rollouts),
            num_tokens=sum(get_seq_len(rollout) for rollout in rollouts),
            num_trainable_rollouts=sum(not rollout.get("is_filtered", False) for rollout in rollouts),
            abs_advantage_mean=mean(abs_advantages) if abs_advantages else 0.0,
            abs_advantage_max=max(abs_advantages) if abs_advantages else 0.0,
        )

    def clone(self) -> "ReplayGroup":
        return ReplayGroup(
            rollouts=copy.deepcopy(self.rollouts),
            env_name=self.env_name,
            policy_step=self.policy_step,
            insert_step=self.insert_step,
            num_rollouts=self.num_rollouts,
            num_tokens=self.num_tokens,
            num_trainable_rollouts=self.num_trainable_rollouts,
            abs_advantage_mean=self.abs_advantage_mean,
            abs_advantage_max=self.abs_advantage_max,
        )


def chunk_rollouts(
    rollouts: list[vf.RolloutOutput],
    rollouts_per_example: int,
) -> list[list[vf.RolloutOutput]]:
    if rollouts_per_example <= 0:
        raise ValueError("rollouts_per_example must be positive")
    if len(rollouts) % rollouts_per_example != 0:
        raise ValueError("Rollouts must be divisible by rollouts_per_example")
    return [rollouts[idx : idx + rollouts_per_example] for idx in range(0, len(rollouts), rollouts_per_example)]


def flatten_rollout_groups(groups: list[list[vf.RolloutOutput]]) -> list[vf.RolloutOutput]:
    flattened: list[vf.RolloutOutput] = []
    for group in groups:
        flattened.extend(group)
    return flattened


def compute_desired_replay_progress(
    batch_target: int,
    replay_fraction: float,
    rollouts_per_example: int,
    use_token_batching: bool,
) -> int:
    if batch_target < 0:
        raise ValueError("batch_target must be non-negative")

    if use_token_batching:
        return int(batch_target * replay_fraction)

    if batch_target % rollouts_per_example != 0:
        raise ValueError("Rollout-mode batch target must be divisible by rollouts_per_example")
    desired_replay_groups = int((batch_target // rollouts_per_example) * replay_fraction)
    return desired_replay_groups * rollouts_per_example


def compute_replay_targets(
    batch_target: int,
    replay_fraction: float,
    available_replay_progress: int,
    rollouts_per_example: int,
    use_token_batching: bool,
) -> tuple[int, int, int]:
    if batch_target < 0:
        raise ValueError("batch_target must be non-negative")
    if available_replay_progress < 0:
        raise ValueError("available_replay_progress must be non-negative")

    desired_replay_progress = compute_desired_replay_progress(
        batch_target=batch_target,
        replay_fraction=replay_fraction,
        rollouts_per_example=rollouts_per_example,
        use_token_batching=use_token_batching,
    )
    actual_replay_progress = min(desired_replay_progress, available_replay_progress)
    fresh_target = batch_target - actual_replay_progress
    return desired_replay_progress, actual_replay_progress, fresh_target


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        max_off_policy_steps: int,
        seed: int | None = None,
    ):
        self.capacity = capacity
        self.max_off_policy_steps = max_off_policy_steps
        self.rng = random.Random(seed)
        self.groups: list[ReplayGroup] = []

    def _is_stale(self, group: ReplayGroup, current_step: int) -> bool:
        return current_step - group.policy_step > self.max_off_policy_steps

    def _eligible_groups(
        self,
        current_step: int,
        exclude_insert_step: int | None = None,
    ) -> list[ReplayGroup]:
        eligible = [group for group in self.groups if not self._is_stale(group, current_step)]
        if exclude_insert_step is not None:
            eligible = [group for group in eligible if group.insert_step != exclude_insert_step]
        return eligible

    @staticmethod
    def _group_progress(group: ReplayGroup, use_token_batching: bool) -> int:
        return group.num_tokens if use_token_batching else group.num_rollouts

    def add(self, groups: list[ReplayGroup]) -> None:
        if self.capacity <= 0 or not groups:
            return

        self.groups.extend(group.clone() for group in groups)
        overflow = len(self.groups) - self.capacity
        if overflow > 0:
            self.groups = self.groups[overflow:]

    def evict_stale(self, current_step: int) -> None:
        self.groups = [group for group in self.groups if not self._is_stale(group, current_step)]

    def available_progress(
        self,
        use_token_batching: bool,
        current_step: int,
        exclude_insert_step: int | None = None,
    ) -> int:
        return sum(
            self._group_progress(group, use_token_batching)
            for group in self._eligible_groups(current_step, exclude_insert_step=exclude_insert_step)
        )

    def sample(
        self,
        target_progress: int,
        use_token_batching: bool,
        current_step: int,
        exclude_insert_step: int | None = None,
    ) -> list[ReplayGroup]:
        if target_progress <= 0:
            return []

        eligible = self._eligible_groups(current_step, exclude_insert_step=exclude_insert_step)
        if not eligible:
            return []

        indices = list(range(len(eligible)))
        self.rng.shuffle(indices)

        sampled: list[ReplayGroup] = []
        progress = 0
        for idx in indices:
            group = eligible[idx]
            sampled.append(group.clone())
            progress += self._group_progress(group, use_token_batching)
            if progress >= target_progress:
                break

        return sampled

    def get_metrics(self, current_step: int) -> dict[str, float]:
        if not self.groups:
            return {
                "replay/size": 0,
                "replay/capacity_utilization": 0.0,
                "replay/age_steps/mean": 0.0,
                "replay/age_steps/max": 0.0,
            }

        ages = [current_step - group.policy_step for group in self.groups]
        capacity_utilization = len(self.groups) / self.capacity if self.capacity > 0 else 0.0
        return {
            "replay/size": len(self.groups),
            "replay/capacity_utilization": capacity_utilization,
            "replay/age_steps/mean": mean(ages),
            "replay/age_steps/max": max(ages),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "groups.jsonl", "w") as f:
            for group in self.groups:
                f.write(json.dumps(asdict(group), default=make_serializable) + "\n")

    def load(self, path: Path) -> None:
        groups_path = path / "groups.jsonl"
        if not groups_path.exists():
            self.groups = []
            return

        with open(groups_path, "r") as f:
            self.groups = [ReplayGroup(**json.loads(line)) for line in f]

        overflow = len(self.groups) - self.capacity
        if overflow > 0:
            self.groups = self.groups[overflow:]
