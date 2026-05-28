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
        last_eval_step_by_env: dict[str, int],
        is_resumed: bool = False,
    ) -> None:
        self.eval_envs = eval_envs
        self.eval_config = eval_config
        # Shared reference with ``Progress.last_eval_step_by_env``; mutated
        # on every fire so the next checkpoint persists the new step.
        self.last_eval_step_by_env = last_eval_step_by_env

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

        # On resume we skip the startup eval; on fresh start the first
        # trigger fires every env (subject to ``skip_first_step``).
        self.first_trigger = not is_resumed

    def trigger(self, step: int) -> list[str]:
        """Fire eligible envs for ``step`` and return their names. The
        per-env duplicate guard (``last_eval_step_by_env``) skips re-firing
        on resume when ``progress.step`` aligns with an env's interval."""
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        fired: list[str] = []
        for name, interval in self.intervals.items():
            last = self.last_eval_step_by_env.get(name, -1)
            if step <= last:
                continue
            if is_first or step % interval == 0:
                fired.append(name)
                self.last_eval_step_by_env[name] = step
        # Round-robin across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates at example granularity. ``try_schedule``'s
        # continue-group branch still keeps each example's group_size
        # rollouts back-to-back, so per-example prefix-cache locality holds.
        iters = [iter(self.examples_by_env[name]) for name in fired]
        for round_examples in zip_longest(*iters):
            for example in round_examples:
                if example is None:
                    continue
                row = dict(example)
                row["eval_step"] = step
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
