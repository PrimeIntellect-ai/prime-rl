"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The policy watcher calls ``trigger(step)`` after each applied policy,
including startup. The dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured.

A standalone eval may stream an infinite taskset through its one epoch: its tasks are
pulled off the env's ``TaskFeed`` as they become ready, until the taskset ends."""

from __future__ import annotations

from collections import deque
from itertools import zip_longest
from typing import TYPE_CHECKING

from prime_rl.orchestrator.types import TaskRequest

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.envs import EvalEnvs
    from prime_rl.orchestrator.task_feed import TaskFeed


class EvalSource:
    """Finite-per-epoch source of eval examples."""

    def __init__(
        self,
        eval_envs: EvalEnvs,
        *,
        intervals: dict[str, int] | None = None,
        skip_first_step: bool = False,
        is_resumed: bool = False,
    ) -> None:
        """``intervals`` is the step interval per env of a training run's online evals;
        a standalone eval has none and fires every env on its one trigger."""
        self.skip_first_step = skip_first_step

        self.tasks_by_env: dict[str, list[vf.Task]] = {}
        self.feeds: dict[str, TaskFeed] = {}
        self.group_sizes: dict[str, int] = {}
        self.intervals: dict[str, int] = {}
        for env in eval_envs:
            self.tasks_by_env[env.name] = list(env.examples)
            if env.feed is not None:
                if intervals is not None:  # evals that repeat per checkpoint need a fixed set
                    raise ValueError(f"Eval env {env.name} has an infinite taskset — set select.limit to bound it")
                self.feeds[env.name] = env.feed
            self.group_sizes[env.name] = env.config.group_size
            self.intervals[env.name] = intervals[env.name] if intervals is not None else 1

        self.queue: deque[TaskRequest] = deque()
        self.streams: dict[str, tuple[int, dict[str, int]]] = {}
        """Fired streaming envs: their step and the rollouts each landed task key still owes."""
        self.owed: dict[str, dict[str, int]] | None = None
        self.groups: dict[str, dict[str, str]] = {}

        # A fresh run evaluates the base policy. Resumed runs apply interval
        # rules to the loaded checkpoint and later policies.
        self.first_trigger = not is_resumed

    def restore(self, owed: dict[str, dict[str, int]], groups: dict[str, dict[str, str]]) -> None:
        """Rollouts the next trigger still owes per env and task key, the rest having
        landed before a resume; a task without an entry is complete. ``groups`` is the
        group id the landed rollouts of a task carry, which the owed ones join."""
        self.owed = owed
        self.groups = groups

    def trigger(self, step: int, *, force: bool = False) -> list[str]:
        """Fire eligible envs for ``step`` and return their names. On resume
        ``first_trigger`` is False, so the startup/base eval doesn't re-run.
        ``force`` fires every env regardless of interval (e.g. the evals process's
        final-checkpoint eval)."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.skip_first_step:
            return []
        fired = [
            name
            for name, interval in self.intervals.items()
            if (is_first or force or step % interval == 0) and (self.tasks_by_env[name] or name in self.feeds)
        ]
        owed, self.owed = self.owed, None
        for name in fired:
            if name in self.feeds:
                self.streams[name] = (step, dict(owed[name]) if owed is not None else {})
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds
        iters = [iter(self.tasks_by_env[name]) for name in fired]
        for round_tasks in zip_longest(*iters):
            for env_name, task in zip(fired, round_tasks, strict=True):
                if task is None:
                    continue
                rollouts = self.group_sizes[env_name]
                if owed is not None:
                    # duplicate tasks share a key: each takes up to a group of what the key owes
                    rollouts = min(rollouts, owed[env_name].get(task.key, 0))
                    owed[env_name][task.key] = owed[env_name].get(task.key, 0) - rollouts
                if rollouts > 0:
                    group_id = self.groups.get(env_name, {}).get(task.key)
                    self.queue.append(
                        TaskRequest(env_name=env_name, task=task, step=step, rollouts=rollouts, group_id=group_id)
                    )
        return fired

    def next_task(self) -> TaskRequest | None:
        """Pop the next eval task, or ``None`` when the queue is empty and no stream has a
        task ready."""
        if self.queue:
            return self.queue.popleft()
        for env_name, (step, owed) in self.streams.items():
            while (task := self.feeds[env_name].poll()) is not None:
                # a key that landed before a resume owes only the rest of its group
                rollouts = owed.pop(task.key, self.group_sizes[env_name])
                if rollouts > 0:
                    group_id = self.groups.get(env_name, {}).get(task.key)
                    return TaskRequest(env_name=env_name, task=task, step=step, rollouts=rollouts, group_id=group_id)
        return None

    def streaming(self, env_name: str) -> bool:
        """Whether ``env_name`` is a fired stream whose taskset has not ended."""
        return env_name in self.streams and not self.feeds[env_name].done

    def cancel_step(self, step: int) -> list[TaskRequest]:
        """Remove and return queued examples for a superseded eval step."""
        cancelled = [request for request in self.queue if request.step == step]
        self.queue = deque(request for request in self.queue if request.step != step)
        return cancelled

    def __bool__(self) -> bool:
        # a stream with no task ready yet still has work until its taskset ends
        return bool(self.queue) or any(self.streaming(name) for name in self.streams)

    def __len__(self) -> int:
        return len(self.queue)
