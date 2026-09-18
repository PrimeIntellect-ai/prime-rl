"""NGU logical visits spanning independent, version-stamped rollout groups."""

from __future__ import annotations

import random
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import verifiers.v1 as vf

from prime_rl.configs.algorithm import NGUAlgoConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import train_work


def dump_episode(episode: vf.Episode) -> dict:
    return episode.model_dump(mode="python", context={"float_decimals": None})


def history_tokens(episode: vf.Episode) -> int:
    return sum(len(node.token_ids) for trace in episode.traces for node in trace.nodes)


@dataclass
class NGUCohort:
    episodes: list[vf.Episode] = field(default_factory=list)
    attempts: int = 0
    successes: int = 0


@dataclass
class Visit:
    task: vf.Task
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cohort: NGUCohort = field(default_factory=NGUCohort)
    rounds: int = 0


class NGUController:
    def __init__(self, config: NGUAlgoConfig, env_name: str):
        self.config = config
        self.env_name = env_name
        self.rng = random.Random(config.seed)
        self.visits: dict[str, Visit] = {}
        self.retries: deque[str] = deque()
        self.counters: Counter[str] = Counter()

    def start(self, task: vf.Task, step: int) -> TaskRequest:
        group_id = str(uuid.uuid4())
        self.visits[group_id] = Visit(task)
        self.counters["visits"] += 1
        return TaskRequest(self.env_name, task, step, group_id=group_id)

    def next_retry(self, step: int) -> TaskRequest:
        group_id = self.retries.popleft()
        self.counters["retry_rounds"] += 1
        return TaskRequest(self.env_name, self.visits[group_id].task, step, group_id=group_id)

    def finish(
        self, group_id: str, episodes: list[vf.Episode], *, complete: bool, min_version: int
    ) -> NGUCohort | None:
        visit = self.visits.pop(group_id)
        self.counters["rounds"] += 1
        pairs = list(iter_trainable_traces(episodes))
        if not complete or any(not e.ok or len(e.traces) != 1 for e in episodes) or len(pairs) != len(episodes):
            self.counters["incomplete_visits"] += 1
            return None
        if any(train_work(e).policy is None for e in episodes):
            raise ValueError("NGU needs live-policy provenance on every episode")
        rewards = [trace.reward for _, trace in pairs]
        if any(reward not in (0, 1) for reward in rewards):
            raise ValueError("NGU requires unshaped binary rewards (0 or 1)")
        if any(e.task.hash != visit.task.hash for e in episodes):
            raise ValueError("NGU round mixed tasks")
        visit.rounds += 1
        visit.cohort.attempts += len(rewards)
        visit.cohort.successes += int(sum(rewards))
        self.counters["valid_attempts"] += len(rewards)
        for episode in episodes:
            trace = episode.traces[0]
            trace.info["ngu_visit"] = visit.id
            trace.info["ngu_round"] = visit.rounds
        visit.cohort.episodes.extend(episodes)
        self._expire(visit, min_version)
        if visit.cohort.successes:
            self.counters["successful_visits"] += 1
            self.counters["first_success_attempts"] += visit.cohort.attempts
            return visit.cohort
        if self.rng.random() >= self.config.continuation_probability:
            self.counters["give_up"] += 1
            return None
        next_id = str(uuid.uuid4())
        self.visits[next_id] = visit
        self.retries.append(next_id)
        for active in self.visits.values():
            self._expire(active, min_version)
        self._bound_history()
        self.counters["continued"] += 1
        return None

    def _expire(self, visit: Visit, min_version: int) -> None:
        kept = [e for e in visit.cohort.episodes if train_work(e).policy.start >= min_version]
        self.counters["expired_payloads"] += len(visit.cohort.episodes) - len(kept)
        visit.cohort.episodes = kept

    def _bound_history(self) -> None:
        tokens = sum(history_tokens(e) for v in self.visits.values() for e in v.cohort.episodes)
        while tokens > self.config.max_history_tokens:
            oldest = min(
                (v for v in self.visits.values() if v.cohort.episodes),
                key=lambda v: train_work(v.cohort.episodes[0]).policy.start,
            )
            tokens -= history_tokens(oldest.cohort.episodes.pop(0))
            self.counters["evicted_payloads"] += 1

    def buffered_episode_ids(self) -> set[str]:
        return {e.id for visit in self.visits.values() for e in visit.cohort.episodes}

    def metrics(self) -> dict[str, float]:
        result = dict(self.counters)
        self.counters.clear()
        result["active_visits"] = len(self.visits)
        result["queued_retries"] = len(self.retries)
        result["history_tokens"] = sum(history_tokens(e) for v in self.visits.values() for e in v.cohort.episodes)
        return {f"ngu/{self.env_name}/{key}": float(value) for key, value in result.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "visits": [
                {
                    "task": visit.task,
                    "id": visit.id,
                    "rounds": visit.rounds,
                    "attempts": visit.cohort.attempts,
                    "successes": visit.cohort.successes,
                    "episodes": [dump_episode(e) for e in visit.cohort.episodes],
                }
                for visit in self.visits.values()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["rng"])
        self.visits.clear()
        self.retries.clear()
        for saved in state["visits"]:
            visit = Visit(
                task=saved["task"],
                id=saved["id"],
                rounds=saved["rounds"],
                cohort=NGUCohort(
                    [vf.WireEpisode.model_validate(e) for e in saved["episodes"]],
                    saved["attempts"],
                    saved["successes"],
                ),
            )
            # Only completed rounds contribute counts. A partially dispatched round restarts fresh.
            group_id = str(uuid.uuid4())
            self.visits[group_id] = visit
            self.retries.append(group_id)
        self.counters["resumed_visits"] += len(self.visits)
