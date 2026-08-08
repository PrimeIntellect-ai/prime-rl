"""EvalSource: trigger-driven, finite-per-epoch pull of eval tasks.

The orchestrator pokes ``trigger(step)`` after each ship + once at
startup; the dispatcher pulls via ``next_task()``
until ``bool(source) == False``. Constructed only when eval is
configured."""

from __future__ import annotations

from collections import deque
from itertools import zip_longest

from prime_rl.configs.orchestrator import EvalConfig
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.types import TaskItem


class EvalSource:
    """Finite-per-epoch source of eval tasks."""

    def __init__(
        self,
        eval_envs: EvalEnvs,
        eval_config: EvalConfig,
        *,
        is_resumed: bool = False,
    ) -> None:
        self.eval_config = eval_config

        self.tasks_by_env: dict[str, list[TaskItem]] = {}
        self.intervals: dict[str, int] = {}
        for env in eval_envs:
            rows = [TaskItem(env_name=env.name, data=data) for data in env.eval_tasks]
            self.tasks_by_env[env.name] = rows
            self.intervals[env.name] = env.config.interval

        self.queue: deque[TaskItem] = deque()

        # On resume we skip the startup eval; on fresh start the first
        # trigger fires every env (subject to ``skip_first_step``)
        self.first_trigger = not is_resumed

    def trigger(self, step: int) -> list[str]:
        """Fire eligible envs for ``step`` and return their names. On resume
        ``first_trigger`` is False, so the startup/base eval doesn't re-run."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        fired: list[str] = []
        for name, interval in self.intervals.items():
            if is_first or step % interval == 0:
                fired.append(name)
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at task granularity. ``try_schedule``'s
        # continue-group branch still keeps each task's group_size
        # episodes back-to-back, so per-task prefix-cache locality holds.
        iters = [iter(self.tasks_by_env[name]) for name in fired]
        for round_tasks in zip_longest(*iters):
            for task in round_tasks:
                if task is None:
                    continue
                self.queue.append(TaskItem(env_name=task.env_name, data=task.data, eval_step=step))
        return fired

    def next_task(self) -> TaskItem | None:
        """Pop the next eval task, if any."""
        if not self.queue:
            return None
        return self.queue.popleft()

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
