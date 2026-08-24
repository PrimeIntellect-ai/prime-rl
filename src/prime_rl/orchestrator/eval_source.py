"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The orchestrator pokes ``trigger(step)`` after each ship + once at
startup; the dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured."""

from __future__ import annotations

from collections import deque
from itertools import zip_longest

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import EvalConfig
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.types import TaskRequest


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
        for env in eval_envs:
            self.tasks_by_env[env.name] = list(env.examples)
            self.intervals[env.name] = env.config.interval

        self.queue: deque[TaskRequest] = deque()

        # On resume we skip the startup eval; on fresh start the first
        # trigger fires every env (subject to ``skip_first_step``)
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
            if is_first or force or step % interval == 0:
                fired.append(name)
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds
        iters = [iter(self.tasks_by_env[name]) for name in fired]
        for round_tasks in zip_longest(*iters):
            for env_name, task in zip(fired, round_tasks, strict=True):
                if task is None:
                    continue
                self.queue.append(TaskRequest(env_name=env_name, task=task, step=step))
        return fired

    def next_task(self) -> TaskRequest | None:
        """Pop the next eval task, or ``None`` when the queue is empty."""
        if not self.queue:
            return None
        return self.queue.popleft()

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
