"""Batcher: accumulates train rollouts until a batch is ready to ship.

Single responsibility. Holds the in-memory buffer of post-pre-filter rollouts;
the orchestrator drives the ``add → ready? → pop`` cycle. No I/O, no
tokenization, no logging, no async — pure data structure.

Eval trajectories are handled by ``EvalCollector``, not here.
"""

from __future__ import annotations

import verifiers as vf

from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.vf_utils import get_seq_len


class Batcher:
    """Holds the rollout buffer until ``batch_size`` (or token budget) is reached.

    ``add()`` runs the pre-batch filters and drops any rollout flagged by an
    enforcing filter — so a zero-advantage group never counts toward the batch.
    Survivors get their filter-annotation state cleared so the downstream
    post-batch filter pass starts from a clean slate.
    """

    def __init__(
        self,
        *,
        batch_size: int | None,
        token_batch_size: int | None,
        pre_filters: list[RolloutFilter],
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.pre_filters = pre_filters
        self.buf: list[vf.RolloutOutput] = []

        # Per-batch detection counters; reset by ``reset_pre_filter_stats``
        # after each pop so the next batch starts fresh.
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    @property
    def buffered_count(self) -> int:
        return len(self.buf)

    def add(self, rollouts: list[vf.RolloutOutput], policy_version: int) -> None:
        """Apply pre-batch filters; survivors go into the buffer.

        ``policy_version`` is stamped on each rollout so downstream metric
        builders can compute per-rollout off-policy lag.
        """
        for r in rollouts:
            r["_policy_version"] = policy_version

        if self.pre_filters:
            apply_filters(self.pre_filters, rollouts)

        for r in rollouts:
            self.pre_filter_seen += 1
            if r.get("is_filtered"):
                self.pre_filter_dropped += 1
                for name, hit in (r.get("filters") or {}).items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean.
            r["filters"] = {}
            r["is_filtered"] = False
            self.buf.append(r)

    def ready(self) -> bool:
        if self.batch_size is not None:
            return len(self.buf) >= self.batch_size
        assert self.token_batch_size is not None
        return sum(get_seq_len(r) for r in self.buf) >= self.token_batch_size

    def pop(self) -> list[vf.RolloutOutput]:
        """Slice one batch off the front of the buffer. Caller must check ``ready()`` first."""
        if self.batch_size is not None:
            batch = self.buf[: self.batch_size]
            self.buf = self.buf[self.batch_size :]
            return batch
        assert self.token_batch_size is not None
        cut = 0
        running = 0
        for i, r in enumerate(self.buf):
            running += get_seq_len(r)
            cut = i + 1
            if running >= self.token_batch_size:
                break
        batch = self.buf[:cut]
        self.buf = self.buf[cut:]
        return batch

    def reset_pre_filter_stats(self) -> None:
        """Clear the per-batch pre-filter counters. Call after the orchestrator
        has emitted them via the metrics builder."""
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
