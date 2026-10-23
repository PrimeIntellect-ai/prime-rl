"""Task sampling balanced by groups with nonzero RL credit sent to training."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler
from prime_rl.orchestrator.utils import episode_group_id

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import InverseContributionSamplerConfig


class InverseContributionSampler(TaskSampler):
    """Weight tasks by inverse retained contribution count, independently of reward."""

    tracks_inflight = True

    def __init__(self, config: InverseContributionSamplerConfig, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        if not isinstance(tasks, Sequence) or not tasks:
            raise ValueError("InverseContributionSampler requires a nonempty finite taskset")
        self.tasks = tuple(tasks)
        keys = [task.key for task in self.tasks]
        if None in keys or len(set(keys)) != len(keys):
            raise ValueError("Task keys must be non-null and unique within a taskset")
        self.exponent = config.exponent
        self.allow_exponent_change_on_resume = config.allow_exponent_change_on_resume
        self.rng = random.Random(config.seed)
        self.counts = dict.fromkeys(keys, 0)
        self.task_by_key = dict(zip(keys, self.tasks, strict=True))
        self.counted_groups: dict[str, str] = {}
        self.fully_solved_group_size = config.fully_solved_group_size
        self.fully_solved_groups: set[str] = set()
        self.cycle_until_first_pass = config.cycle_until_first_pass
        self.solved: set[str] = set()
        self.cycle: deque[str] = deque()
        self.initial_counts_path = config.initial_counts_path
        self.initial_solved_path = config.initial_solved_path
        if self.initial_counts_path is not None:
            initial = json.loads(self.initial_counts_path.read_text())
            self._validate_counts(initial, require_all=False)
            self.counts.update(initial)
        if self.initial_solved_path is not None:
            initial_solved = json.loads(self.initial_solved_path.read_text())
            self._validate_solved(initial_solved)
            self.solved = set(initial_solved)
        initial = [key for key in keys if key not in self.solved] if self.cycle_until_first_pass else []
        self.rng.shuffle(initial)
        self.first_cycle = deque(initial)

    def _validate_solved(self, keys: list[str]) -> None:
        if not isinstance(keys, list) or any(not isinstance(key, str) or key not in self.counts for key in keys):
            raise ValueError("Solved tasks must be a list of known task keys")
        if len(set(keys)) != len(keys):
            raise ValueError("Solved task keys must be unique")

    def _validate_counts(self, counts: dict[str, int], *, require_all: bool) -> None:
        if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("Contribution counts must be a mapping of task keys to nonnegative integers")
        if set(counts) - self.counts.keys() or (require_all and set(counts) != self.counts.keys()):
            raise ValueError("Contribution checkpoint task keys do not match this taskset")

    def probabilities(self) -> dict[str, float]:
        weights = {task.key: (1 + self.counts[task.key]) ** -self.exponent for task in self.tasks}
        total = sum(weights.values())
        return {key: weight / total for key, weight in weights.items()}

    def __next__(self) -> vf.Task:
        task = self.next_available(set())
        assert task is not None
        return task

    def _eligible(self, excluded: set[str]) -> tuple[list[str], list[str]]:
        if self.first_cycle:
            return [key for key in self.first_cycle if key not in excluded], []
        unsolved = [key for key in self.counts if key not in self.solved] if self.cycle_until_first_pass else []
        if not unsolved:
            return [], [key for key in self.counts if key not in excluded]
        pool = list(self.cycle) if self.cycle else unsolved
        return [key for key in pool if key not in excluded], [
            key for key in self.counts if key in self.solved and key not in excluded
        ]

    def has_available(self, excluded: set[str]) -> bool:
        pool, individual = self._eligible(excluded)
        return bool(pool or individual)

    def next_available(self, excluded: set[str]) -> vf.Task | None:
        eligible_pool, eligible_individual = self._eligible(excluded)
        if not eligible_pool and not eligible_individual:
            return None
        if self.first_cycle:
            key = eligible_pool[0]
            self.first_cycle.remove(key)
            return self.task_by_key[key]
        probs = self.probabilities()
        if not eligible_pool:
            key = self.rng.choices(eligible_individual, weights=[probs[key] for key in eligible_individual], k=1)[0]
            return self.task_by_key[key]
        pool = self.rng.choices(
            [None, *eligible_individual],
            weights=[
                sum(probs[key] for key in probs if key not in self.solved and key not in excluded),
                *(probs[key] for key in eligible_individual),
            ],
            k=1,
        )[0]
        if pool is not None:
            return self.task_by_key[pool]
        if not self.cycle:
            unsolved = [key for key in probs if key not in self.solved]
            self.rng.shuffle(unsolved)
            self.cycle.extend(unsolved)
        key = next(key for key in self.cycle if key not in excluded)
        self.cycle.remove(key)
        return self.task_by_key[key]

    def requeue_abandoned(self, key: str, initial_sweep: bool) -> None:
        if key not in self.counts:
            raise ValueError(f"Unknown abandoned task key: {key}")
        if not self.cycle_until_first_pass or key in self.solved:
            return
        queue = self.first_cycle if initial_sweep and self.first_cycle else self.cycle
        if key not in queue:
            queue.append(key)

    def observe_episode(self, episode: vf.Episode) -> None:
        if not episode.ok or not any(
            not trace.has_error and trace.agent.trainable and trace.num_turns > 0 and trace.reward == 1.0
            for trace in episode.traces
        ):
            return
        key = episode.task.key
        if key not in self.counts:
            raise ValueError(f"Unknown solved task key: {key}")
        self.solved.add(key)
        self.first_cycle = deque(task_key for task_key in self.first_cycle if task_key != key)
        self.cycle = deque(task_key for task_key in self.cycle if task_key != key)

    def observe(self, group: list[vf.Episode]) -> None:
        for episode in group:
            self.observe_episode(episode)
        if self.fully_solved_group_size is None or len(group) != self.fully_solved_group_size:
            return
        # Failed/cancelled members are omitted by the sink. Require a full,
        # distinct cohort, not merely all successful survivors of a group.
        if len({episode.id for episode in group}) != len(group):
            return
        task_keys = {episode.task.key for episode in group}
        group_ids = {episode_group_id(episode) for episode in group}
        if len(task_keys) != 1 or len(group_ids) != 1:
            raise ValueError("Fully solved group must have one task key and group ID")
        if not all(
            episode.traces
            and all(not trace.has_error for trace in episode.traces)
            and any(trace.agent.trainable for trace in episode.traces)
            and all(trace.reward == 1.0 for trace in episode.traces if trace.agent.trainable)
            for episode in group
        ):
            return
        task_key = group[0].task.key
        group_id = episode_group_id(group[0])
        previous = self.counted_groups.get(group_id)
        if previous is not None and previous != task_key:
            raise ValueError(f"Counted group {group_id} contains multiple task keys")
        if previous is None:
            self.counts[task_key] += 1
            self.counted_groups[group_id] = task_key
            self.fully_solved_groups.add(group_id)

    def on_train_batch(self, episodes: list[vf.Episode]) -> None:
        groups: dict[str, str] = {}
        for episode in episodes:
            # The caller supplies only the shipped cohort. Zero-credit padding must not
            # count; negative advantages count just as much as positive advantages.
            contributes = any(
                not trace.has_error
                and trace.agent.trainable
                and any(
                    node.sampled and any(node.mask) and math.isfinite(advantage) and advantage != 0.0
                    for node in trace.nodes
                    for advantage in node.advantages or []
                )
                for trace in episode.traces
            )
            if not contributes:
                continue
            task_key = episode.task.key
            if task_key not in self.counts:
                raise ValueError(f"Unknown contributing task key: {task_key}")
            group_id = episode_group_id(episode)
            previous = groups.get(group_id, self.counted_groups.get(group_id))
            if previous is not None and previous != task_key:
                raise ValueError(f"Contributing group {group_id} contains multiple task keys")
            groups[group_id] = task_key
        for group_id, task_key in groups.items():
            if group_id not in self.counted_groups:
                self.counts[task_key] += 1
                self.counted_groups[group_id] = task_key

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "inverse_contribution",
            "version": 3,
            "fully_solved_group_size": self.fully_solved_group_size,
            "fully_solved_groups": sorted(self.fully_solved_groups),
            "exponent": self.exponent,
            "rng": self.rng.getstate(),
            "counts": dict(self.counts),
            "task_order": list(self.task_by_key),
            "counted_groups": dict(self.counted_groups),
            "cycle_until_first_pass": self.cycle_until_first_pass,
            "solved": sorted(self.solved),
            "cycle": list(self.cycle),
            "first_cycle": list(self.first_cycle),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) == {"cursor"} and self.initial_counts_path is not None:
            # A standard sampler contains no contribution history. Migration requires
            # an explicit seed built from the retained checkpoint's shipped cohorts.
            if self.cycle_until_first_pass and self.initial_solved_path is None:
                raise ValueError(
                    "Cycling migration requires initial_solved_path; contributions do not imply full passes"
                )
            return
        if state_dict.get("type") != "inverse_contribution" or state_dict.get("version") not in (2, 3):
            raise ValueError("Expected inverse-contribution state; standard migration needs initial_counts_path")
        if state_dict["exponent"] != self.exponent and not self.allow_exponent_change_on_resume:
            raise ValueError("Sampler exponent differs from the checkpoint")
        if state_dict["cycle_until_first_pass"] != self.cycle_until_first_pass:
            raise ValueError("Cycling mode differs from the checkpoint")
        counts = state_dict["counts"]
        self._validate_counts(counts, require_all=True)
        if state_dict["task_order"] != list(self.task_by_key):
            raise ValueError("Task order differs from the checkpoint; deterministic sampling requires the same order")
        groups = state_dict["counted_groups"]
        perfect_groups = state_dict.get("fully_solved_groups", [])
        if state_dict["version"] == 3 and state_dict["fully_solved_group_size"] != self.fully_solved_group_size:
            raise ValueError("Fully solved group size differs from checkpoint")
        if len(set(perfect_groups)) != len(perfect_groups) or not set(perfect_groups) <= groups.keys():
            raise ValueError("Invalid fully solved group history")
        if any(key not in self.counts for key in groups.values()):
            raise ValueError("Counted groups contain unknown task keys")
        if any(n > counts[key] for key, n in Counter(groups.values()).items()):
            raise ValueError("Counted groups exceed contribution counts")
        self._validate_solved(state_dict["solved"])
        cycle = state_dict["cycle"]
        self._validate_solved(cycle)
        first_cycle = state_dict["first_cycle"]
        self._validate_solved(first_cycle)
        if first_cycle and (cycle or not self.cycle_until_first_pass or set(first_cycle) & set(state_dict["solved"])):
            raise ValueError("Initial sweep conflicts with solved tasks or later cycling state")
        if set(cycle) & set(state_dict["solved"]) or (cycle and not self.cycle_until_first_pass):
            raise ValueError("Cycle contains solved tasks or cycling is disabled")
        self.rng.setstate(state_dict["rng"])
        self.counts = dict(counts)
        self.counted_groups = dict(groups)
        self.fully_solved_groups = set(perfect_groups)
        self.solved = set(state_dict["solved"])
        self.cycle = deque(cycle)
        self.first_cycle = deque(first_cycle)

    def metrics(self) -> dict[str, float]:
        values = list(self.counts.values())
        perfect_counts = Counter(self.counted_groups[gid] for gid in self.fully_solved_groups)
        contribution_values = [n - perfect_counts[key] for key, n in self.counts.items()]
        distribution = self.probabilities()
        unsolved = [key for key in distribution if key not in self.solved]
        if self.first_cycle:
            distribution = dict.fromkeys(self.first_cycle, 1.0 / len(self.first_cycle))
        elif self.cycle_until_first_pass and unsolved:
            pool_share = sum(distribution[key] for key in unsolved)
            distribution.update(dict.fromkeys(unsolved, pool_share / len(unsolved)))
        probs = list(distribution.values())
        return {
            "contributions/total": float(sum(contribution_values)),
            "contributions/unseen": float(contribution_values.count(0)),
            "contributions/max": float(max(contribution_values)),
            "sampling/counted_visits": float(sum(values)),
            "sampling/fully_solved_groups": float(len(self.fully_solved_groups)),
            "sampling/effective_tasks": 1.0 / sum(p * p for p in probs),
            "sampling/never_solved": float(len(self.counts) - len(self.solved)),
        }
