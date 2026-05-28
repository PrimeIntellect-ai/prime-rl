"""EvalSource: trigger-driven finite-per-epoch pull of eval examples.

Holds the per-env example lists + intervals + the pending queue. The
orchestrator pokes it via ``trigger(step)`` after each ship step (and
once at startup), and the dispatcher pulls one example at a time via
``next_example(available_permits)`` until ``bool(source) == False``.

The dispatcher still owns scheduling priority (``dispatcher.DispatcherMode``)
and capacity (``max_inflight`` counter); this source owns the per-env
permit cost lookup. Mirrors ``TrainSource.next_example`` so the
dispatcher hits both sources through a single symmetric API.

Constructed only when eval is configured — the orchestrator gates
construction on ``config.eval is not None``, so this class can take
non-Optional ``eval_envs`` / ``eval_config`` and the body is free of
"is eval even configured?" branches.
"""

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
        # Shared reference with ``Progress.last_eval_step_by_env`` — we
        # mutate it on every fire so the orchestrator's next checkpoint
        # save persists the new step. On resume, the orchestrator hands
        # us back the loaded dict.
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

        # Pending eval examples in FIFO order, each carrying ``env_name`` +
        # ``eval_step`` baked in. ``trigger`` round-robins across fired
        # envs at example granularity, copying each row fresh so the same
        # row can be enqueued at multiple eval steps over the run without
        # aliasing.
        self.queue: deque[dict] = deque()

        # ``first_trigger`` controls the startup-eval semantics: on a
        # fresh start (``is_resumed=False``) the first ``trigger()`` call
        # fires every env unconditionally (subject to ``skip_first_step``).
        # On a resume we skip the startup eval entirely so the user
        # doesn't get a duplicate baseline eval at the resume step.
        self.first_trigger = not is_resumed

    # ── trigger ────────────────────────────────────────────────────────────

    def trigger(self, step: int) -> list[str]:
        """Fire eligible envs for ``step`` and return their names.

        First call on a fresh start (not a resume): fire every env, unless
        ``skip_first_step`` is True. Subsequent calls: fire each env iff
        ``step % interval == 0``. Per-env duplicate guard:
        ``last_eval_step_by_env`` records every fire and skips re-firing
        the same env at the same (or earlier) step, which would otherwise
        happen on resume when ``progress.step`` aligns with the env's
        interval. Caller (``Orchestrator``) only ever invokes this with
        monotonically increasing ``step`` values.
        """
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        fired: list[str] = []
        for name, interval in self.intervals.items():
            last = self.last_eval_step_by_env.get(name, -1)
            if step <= last:
                continue  # already fired at >= this step pre-resume
            if is_first or step % interval == 0:
                fired.append(name)
                self.last_eval_step_by_env[name] = step
        # Round-robin enqueue across fired envs (A₁, B₁, A₂, B₂, …) so the
        # dispatcher rotates through them at example granularity instead of
        # draining all of A before starting B. ``try_schedule``'s "continue
        # existing group" branch still keeps each example's ``group_size``
        # rollouts back-to-back, so per-example prefix-cache locality is
        # intact. ``zip_longest`` pads short envs with ``None`` once they
        # run out — those are skipped.
        iters = [iter(self.examples_by_env[name]) for name in fired]
        for round_examples in zip_longest(*iters):
            for example in round_examples:
                if example is None:
                    continue
                row = dict(example)
                row["eval_step"] = step
                self.queue.append(row)
        return fired

    # ── pull ───────────────────────────────────────────────────────────────

    def next_example(self, available_permits: int) -> dict | None:
        """Pop the next eval example if the dispatcher can afford its cost.

        Returns ``None`` when the queue is empty or the head requires more
        permits than available (head stays put — the dispatch loop will
        retry on the next iteration once permits free up).
        """
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
