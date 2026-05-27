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

from prime_rl.configs.orchestrator import EvalConfig
from prime_rl.orchestrator.envs import EvalEnvs


class EvalSource:
    """Finite-per-epoch source of eval examples."""

    def __init__(self, eval_envs: EvalEnvs, eval_config: EvalConfig) -> None:
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

        # Pending eval examples in FIFO order, each carrying ``env_name`` +
        # ``_eval_step`` baked in. ``trigger`` extends with one fresh copy
        # per example per fired env, so the same row can be enqueued at
        # multiple eval steps over the run without aliasing.
        self.queue: deque[dict] = deque()

        # The first ``trigger`` call (orchestrator startup) fires every
        # env unconditionally — the startup eval evaluates whatever model
        # state we begin from (base model or resumed checkpoint) before
        # any train rollouts. ``eval.skip_first_step`` gates this one
        # call; subsequent calls use the usual ``% interval`` gating.
        self.first_trigger = True

    # ── trigger ────────────────────────────────────────────────────────────

    def trigger(self, step: int) -> list[str]:
        """Fire eligible envs for ``step`` and return their names.

        First call (startup): fire every env, unless ``skip_first_step``
        is True. Subsequent calls: fire each env iff ``step % interval ==
        0``. Caller (``Orchestrator``) only ever invokes this with
        monotonically increasing ``step`` values, so no double-fire guard
        is needed.
        """
        is_first, self.first_trigger = self.first_trigger, False
        if is_first and self.eval_config.skip_first_step:
            return []
        fired = [name for name, interval in self.intervals.items() if is_first or step % interval == 0]
        for name in fired:
            self.enqueue(name, step)
        return fired

    def enqueue(self, env_name: str, step: int) -> None:
        for example in self.examples_by_env.get(env_name, []):
            row = dict(example)
            row["_eval_step"] = step
            self.queue.append(row)

    # ── pull ───────────────────────────────────────────────────────────────

    def next_example(self, available_permits: int) -> dict | None:
        """Pop the next eval example if the dispatcher can afford its cost.

        Returns ``None`` when the queue is empty or the head requires more
        permits than available (head stays put — the dispatch loop will
        retry on the next iteration once permits free up).

        Mirrors ``TrainSource.next_example``: one call commits, no
        separate peek/pop pair.
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
