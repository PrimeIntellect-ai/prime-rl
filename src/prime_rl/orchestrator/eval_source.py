"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The policy watcher calls ``trigger(step)`` after each applied policy,
including startup. The dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from itertools import zip_longest
from typing import TYPE_CHECKING

from prime_rl.orchestrator.types import TaskRequest

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.configs.orchestrator import EvalConfig
    from prime_rl.orchestrator.envs import EvalEnvs


class EvalSource:
    """Finite-per-epoch source of eval examples."""

    def __init__(
        self,
        eval_envs: EvalEnvs,
        eval_config: EvalConfig,
        *,
        is_resumed: bool = False,
    ) -> None:
        self.eval_envs = eval_envs
        self.eval_config = eval_config

        self.tasks_by_env: dict[str, list[vf.Task]] = {}
        self.intervals: dict[str, int] = {}
        self.group_sizes: dict[str, int] = {}
        for env in eval_envs:
            self.tasks_by_env[env.name] = list(env.examples)
            self.intervals[env.name] = env.config.interval
            self.group_sizes[env.name] = env.config.group_size

        self.queue: deque[TaskRequest] = deque()
        self.cursor = 0
        self._next_index = 0
        self._completed: set[int] = set()
        self.partial: dict[int, int] = {}
        self.group_ids: dict[int, str] = {}
        self.revision = 0
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [
                    (name, self.group_sizes[name], [(task.key, task.hash) for task in tasks])
                    for name, tasks in self.tasks_by_env.items()
                ]
            ).encode()
        ).hexdigest()
        self._triggered_task_counts: dict[tuple[str, int], int] = {}
        self._triggered_rollout_counts: dict[tuple[str, int], int] = {}

        # A fresh run evaluates the base policy. Resumed runs apply interval
        # rules to the loaded checkpoint and later policies.
        self.first_trigger = not is_resumed

    def trigger(self, step: int, *, force: bool = False) -> list[str]:
        """Fire eligible envs for ``step`` and return their names. On resume
        ``first_trigger`` is False, so the startup/base eval doesn't re-run.
        ``force`` fires every env regardless of interval (e.g. the evals process's
        final-checkpoint eval)."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        fired: list[str] = []
        for name, interval in self.intervals.items():
            if (is_first or force or step % interval == 0) and self.tasks_by_env[name]:
                fired.append(name)
        queued_counts: Counter[str] = Counter()
        rollout_counts: Counter[str] = Counter()
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds
        iters = [iter(self.tasks_by_env[name]) for name in fired]
        for round_tasks in zip_longest(*iters):
            for env_name, task in zip(fired, round_tasks, strict=True):
                if task is None:
                    continue
                source_index = self._next_index
                self._next_index += 1
                if source_index < self.cursor or source_index in self._completed:
                    continue
                remaining = self.group_sizes[env_name] - self.partial.get(source_index, 0)
                if remaining <= 0:
                    raise ValueError(f"Invalid partial rollout count for task {source_index}")
                self.queue.append(
                    TaskRequest(
                        env_name=env_name, task=task, step=step, source_index=source_index, num_rollouts=remaining
                    )
                )
                queued_counts[env_name] += 1
                rollout_counts[env_name] += remaining
        for env_name, count in queued_counts.items():
            self._triggered_task_counts[(env_name, step)] = count
            self._triggered_rollout_counts[(env_name, step)] = rollout_counts[env_name]
        return [name for name in fired if queued_counts[name]]

    def triggered_task_count(self, env_name: str, step: int) -> int:
        return self._triggered_task_counts.get((env_name, step), 0)

    def triggered_rollout_count(self, env_name: str, step: int) -> int:
        return self._triggered_rollout_counts.get((env_name, step), 0)

    def record_attempt(self, source_index: int, group_size: int, group_id: str | None = None) -> None:
        if source_index < self.cursor or source_index in self._completed:
            raise ValueError(f"Task {source_index} is already complete")
        count = self.partial.get(source_index, 0) + 1
        if count > group_size:
            raise ValueError(f"Too many completed attempts for task {source_index}")
        if group_id is not None:
            if self.group_ids.setdefault(source_index, group_id) != group_id:
                raise ValueError(f"Group identity changed for task {source_index}")
        self.revision += 1
        if count == group_size:
            self.partial.pop(source_index, None)
            self.group_ids.pop(source_index, None)
            self.mark_completed(source_index)
        else:
            self.partial[source_index] = count

    def mark_completed(self, source_index: int) -> bool:
        """Advance the durable cursor only across a fully completed prefix."""
        if source_index < self.cursor:
            return False
        self._completed.add(source_index)
        previous = self.cursor
        while self.cursor in self._completed:
            self._completed.remove(self.cursor)
            self.cursor += 1
        return self.cursor != previous

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "completed": sorted(self._completed),
            "partial": self.partial.copy(),
            "group_ids": self.group_ids.copy(),
            "fingerprint": self.fingerprint,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not {"cursor"} <= set(state_dict) <= {"cursor", "completed", "partial", "group_ids", "fingerprint"}:
            raise ValueError("Invalid eval source checkpoint fields")
        cursor = state_dict["cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError(f"Eval source checkpoint cursor must be a non-negative integer, got {cursor!r}")
        self.cursor = cursor
        if state_dict.get("fingerprint", self.fingerprint) != self.fingerprint:
            raise ValueError("Eval dataset order, task contents, or group sizes changed since checkpoint")
        completed = state_dict.get("completed", [])
        partial = state_dict.get("partial", {})
        if any(type(index) is not int or index < cursor for index in [*completed, *partial]):
            raise ValueError("Invalid eval checkpoint task index")
        if any(type(count) is not int or count < 1 for count in partial.values()) or set(completed) & set(partial):
            raise ValueError("Invalid eval checkpoint partial counts")
        self._completed = set(completed)
        self.partial = dict(partial)
        self.group_ids = dict(state_dict.get("group_ids", {}))
        if not set(self.group_ids) <= set(partial) or any(
            not isinstance(gid, str) or not gid for gid in self.group_ids.values()
        ):
            raise ValueError("Invalid eval checkpoint group identities")
        self.revision = 0

    def next_task(self) -> TaskRequest | None:
        """Pop the next eval task, or ``None`` when the queue is empty."""
        if not self.queue:
            return None
        return self.queue.popleft()

    def cancel_step(self, step: int) -> list[TaskRequest]:
        """Remove and return queued examples for a superseded eval step."""
        cancelled = [request for request in self.queue if request.step == step]
        self.queue = deque(request for request in self.queue if request.step != step)
        return cancelled

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
