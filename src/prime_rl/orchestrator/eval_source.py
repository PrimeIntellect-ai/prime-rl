"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The policy watcher calls ``trigger(step)`` after each applied policy,
including startup. The dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured."""

from __future__ import annotations

from collections import Counter, deque
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
        self.completed: Counter[tuple[str, int, str, str]] = Counter()

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

    @staticmethod
    def task_key(env_name: str, task: vf.Task, step: int) -> tuple[str, int, str, str]:
        return env_name, step, task.key, task.hash

    def mark_completed(self, env_name: str, task_key: str, task_hash: str, step: int) -> None:
        self.completed[(env_name, step, task_key, task_hash)] += 1

    def state_dict(self) -> dict:
        return {"completed": dict(self.completed)}

    def load_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != {"completed"}:
            raise ValueError("Eval source checkpoint must contain only completed task groups")
        self.completed = Counter(state_dict["completed"])
        completed_steps = {key[1] for key, count in self.completed.items() if count}
        if len(completed_steps) != 1:
            raise ValueError("Eval source checkpoint must contain completed groups from exactly one eval step")
        step = completed_steps.pop()
        for env_name, tasks in self.tasks_by_env.items():
            matched = Counter()
            remaining = []
            for task in tasks:
                key = self.task_key(env_name, task, step)
                if matched[key] < self.completed[key]:
                    matched[key] += 1
                else:
                    remaining.append(task)
            self.tasks_by_env[env_name] = remaining

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
