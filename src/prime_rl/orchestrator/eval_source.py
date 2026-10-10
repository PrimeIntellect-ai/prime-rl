"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The policy watcher calls ``trigger(step)`` after each applied policy,
including startup. The dispatcher pulls via ``next_task()`` until
``bool(source) == False``. Constructed only when eval is configured."""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from collections.abc import Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING

import verifiers.v1 as vf
from verifiers.v1.utils.eval import plan_rollouts

from prime_rl.orchestrator.types import TaskRequest

if TYPE_CHECKING:
    from prime_rl.orchestrator.envs import EvalEnvs


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
        self.group_sizes: dict[str, int] = {}
        self.intervals: dict[str, int] = {}
        for env in eval_envs:
            self.tasks_by_env[env.name] = list(env.examples)
            self.group_sizes[env.name] = env.config.group_size
            self.intervals[env.name] = intervals[env.name] if intervals is not None else 1

        self.queue: deque[TaskRequest] = deque()

        # A fresh run evaluates the base policy. Resumed runs apply interval
        # rules to the loaded checkpoint and later policies.
        self.first_trigger = not is_resumed

    def trigger(
        self, step: int, *, force: bool = False, completed: Iterable[vf.WireEpisode] = ()
    ) -> tuple[list[str], list[vf.WireEpisode]]:
        """Fire eligible envs and return their names and matched saved episodes. On resume
        ``first_trigger`` is False, so the startup/base eval doesn't re-run.
        ``force`` fires every env regardless of interval (e.g. the evals process's
        final-checkpoint eval)."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.skip_first_step:
            return [], []
        fired = [
            name
            for name, interval in self.intervals.items()
            if (is_first or force or step % interval == 0) and self.tasks_by_env[name]
        ]
        saved: dict[str, list[vf.WireEpisode]] = defaultdict(list)
        for episode in completed:
            saved[episode.env.name or episode.env.id].append(episode)
        restored: list[vf.WireEpisode] = []
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds
        iters = [iter(plan_rollouts(self.tasks_by_env[name], self.group_sizes[name], saved[name])) for name in fired]
        for round_tasks in zip_longest(*iters):
            for env_name, planned in zip(fired, round_tasks, strict=True):
                if planned is None:
                    continue
                task, kept, rollouts = planned
                # Each selected occurrence owns one group, including its restored episodes.
                group = vf.GroupInfo(id=str(uuid.uuid4()))
                for episode in kept:
                    episode.group = group
                restored.extend(kept)
                if rollouts > 0:
                    self.queue.append(
                        TaskRequest(env_name=env_name, task=task, step=step, rollouts=rollouts, group_id=group.id)
                    )
        return fired, restored

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
