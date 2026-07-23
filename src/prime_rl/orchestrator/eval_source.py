"""EvalSource: trigger-driven, finite-per-epoch pull of eval examples.

The orchestrator pokes ``trigger(step)`` after each ship + once at
startup; the dispatcher pulls via ``next_example(available_permits)``
until ``bool(source) == False``. Constructed only when eval is
configured."""

from __future__ import annotations

from collections import deque
from itertools import zip_longest

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
        resume_step: int | None = None,
    ) -> None:
        self.eval_envs = eval_envs
        self.eval_config = eval_config

        self.examples_by_env: dict[str, list[dict]] = {}
        self.intervals: dict[str, int] = {}
        for env in eval_envs:
            rows: list[dict] = []
            for ex in env.examples:
                row = dict(ex)
                row["env_name"] = env.name
                rows.append(row)
            self.examples_by_env[env.name] = rows
            self.intervals[env.name] = env.config.interval

        self.queue: deque[dict] = deque()

        # On fresh start the first trigger fires every env (subject to
        # ``skip_first_step``). On resume the first trigger fires only envs
        # whose interval divides the checkpoint step, so an eval that was due
        # at the checkpoint is recovered instead of deferred to the next
        # interval boundary.
        self.is_resumed = is_resumed
        self.resume_step = resume_step
        self.first_trigger = True

    def trigger(self, step: int) -> list[str]:
        """Fire eligible envs for ``step`` and return their names.

        On fresh start the first trigger fires every env (subject to
        ``skip_first_step``). On resume the first trigger fires only envs
        whose interval divides the checkpoint step, recovering an eval that
        was due at the checkpoint instead of deferring to the next boundary.
        """
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        # On resume, evaluate against the checkpoint step (not progress.step,
        # which is checkpoint + 1) so interval-aligned evals are recovered.
        eval_step = self.resume_step if (is_first and self.is_resumed) else step
        fired: list[str] = []
        for name, interval in self.intervals.items():
            if is_first and not self.is_resumed:
                fired.append(name)
            elif eval_step % interval == 0:
                fired.append(name)
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds
        iters = [iter(self.examples_by_env[name]) for name in fired]
        for round_examples in zip_longest(*iters):
            for example in round_examples:
                if example is None:
                    continue
                row = dict(example)
                row["eval_step"] = eval_step
                self.queue.append(row)
        return fired

    def next_example(self, available_permits: int) -> dict | None:
        """Pop the next eval example if the head's permit cost fits in
        ``available_permits``; otherwise leave it for a later call."""
        if not self.queue:
            return None
        head = self.queue[0]
        env = self.eval_envs.get(head["env_name"])
        cost = env.config.group_size if env.requires_group_scoring else 1
        if cost > available_permits:
            return None
        return self.queue.popleft()

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
