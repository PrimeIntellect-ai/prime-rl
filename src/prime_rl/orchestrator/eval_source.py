"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The policy watcher calls ``trigger(step)`` after each applied policy,
including startup. The dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured."""

from __future__ import annotations

from collections import Counter, deque
from itertools import zip_longest
from typing import TYPE_CHECKING

from prime_rl.orchestrator.types import TaskRequest

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.envs import EvalEnvs


class EvalSource:
    """Finite-per-epoch source of eval examples."""

    def __init__(
        self,
        eval_envs: EvalEnvs,
        *,
        skip_first_step: bool = False,
        is_resumed: bool = False,
    ) -> None:
        self.eval_envs = eval_envs
        self.skip_first_step = skip_first_step

        self.tasks_by_env: dict[str, list[vf.Task]] = {}
        self.intervals: dict[str, int] = {}
        for env in eval_envs:
            self.tasks_by_env[env.name] = list(env.examples)
            self.intervals[env.name] = env.config.interval

        self.queue: deque[TaskRequest] = deque()
        self.cursor = 0
        self._next_index = 0
        self._completed: set[int] = set()
        self._triggered_task_counts: dict[tuple[str, int], int] = {}

        # A fresh run evaluates the base policy. Resumed runs apply interval
        # rules to the loaded checkpoint and later policies.
        self.first_trigger = not is_resumed

    def trigger(self, step: int, *, force: bool = False) -> list[str]:
        """Fire eligible envs for ``step`` and return their names. On resume
        ``first_trigger`` is False, so the startup/base eval doesn't re-run.
        ``force`` fires every env regardless of interval (e.g. the evals process's
        final-checkpoint eval)."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.skip_first_step:
            return []
        fired: list[str] = []
        for name, interval in self.intervals.items():
            if (is_first or force or step % interval == 0) and self.tasks_by_env[name]:
                fired.append(name)
        queued_counts: Counter[str] = Counter()
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
                if source_index < self.cursor:
                    continue
                self.queue.append(TaskRequest(env_name=env_name, task=task, step=step, source_index=source_index))
                queued_counts[env_name] += 1
        for env_name, count in queued_counts.items():
            self._triggered_task_counts[(env_name, step)] = count
        return [name for name in fired if queued_counts[name]]

    def triggered_task_count(self, env_name: str, step: int) -> int:
        return self._triggered_task_counts.get((env_name, step), 0)

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
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != {"cursor"}:
            raise ValueError("Eval source checkpoint must contain only a cursor")
        cursor = state_dict["cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError(f"Eval source checkpoint cursor must be a non-negative integer, got {cursor!r}")
        self.cursor = cursor
        self._completed.clear()

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
