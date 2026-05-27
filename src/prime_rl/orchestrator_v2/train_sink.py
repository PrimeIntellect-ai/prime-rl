"""TrainSink: groups train rollouts, computes advantages, applies pre-filters,
and buffers survivors until the batch is ready.

Triggered by ``Rollout.is_group_complete=True``: at that point the
``(env_name, example_id)`` GRPO group has all its surviving rollouts in.
The sink runs ``compute_advantages`` over the group, then ``apply_filters``
for the pre-batch filter pass, then drops filtered rollouts and extends the
batch buffer with the survivors.

Flush signal: ``batch_ready()`` returns True when the buffer has reached
``batch_size`` rollouts (or ``token_batch_size`` tokens). The orchestrator
polls this and calls ``pop()`` to dequeue one batch.

No I/O, no async, no step tracking — pure data structure + signal.
"""

from __future__ import annotations

from collections import defaultdict

import verifiers as vf

from prime_rl.configs.orchestrator import AdvantageConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.orchestrator_v2.dispatcher import Rollout


class TrainSink:
    """Train-side rollout sink. Group → advantages → pre-filter → batch buffer."""

    def __init__(
        self,
        *,
        batch_size: int | None,
        token_batch_size: int | None,
        advantage_config: AdvantageConfig | None,
        pre_filters: list[RolloutFilter],
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.advantage_config = advantage_config
        self.pre_filters = pre_filters

        # In-progress GRPO groups keyed by (env_name, example_id). Each group
        # holds the surviving ``Rollout``\\ s as they arrive; finalized on
        # ``is_group_complete=True``.
        self.pending: dict[tuple[str, int], list[Rollout]] = defaultdict(list)

        # Survivors of the pre-filter pass — waiting to ship.
        self.batch_buf: list[vf.RolloutOutput] = []

        # Per-batch pre-filter detection counters; the orchestrator reads them
        # via ``pre_filter_stats()`` and resets via ``reset_pre_filter_stats()``.
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    # ── ingest ────────────────────────────────────────────────────────────

    def add(self, rollout: Rollout) -> None:
        """Buffer one rollout; finalize the GRPO group on ``is_group_complete=True``."""
        assert rollout.kind == "train", "TrainSink only handles train rollouts"
        key = (rollout.env_name, rollout.example_id)
        self.pending[key].append(rollout)
        if rollout.is_group_complete:
            self.finalize_group(key)

    def finalize_group(self, key: tuple[str, int]) -> None:
        group = self.pending.pop(key, [])
        if not group:
            return
        raws = [r.raw for r in group]
        # Stamp per-rollout policy version (used by the metrics builder).
        for raw, r in zip(raws, group):
            raw["_policy_version"] = r.policy_version

        # Advantages over the surviving rollouts (possibly partial group).
        if self.advantage_config is not None:
            compute_advantages(raws, self.advantage_config)
        else:
            for raw in raws:
                raw["advantage"] = raw.get("reward", 0.0)

        # Pre-batch filter pass.
        if self.pre_filters:
            apply_filters(self.pre_filters, raws)
        for raw in raws:
            self.pre_filter_seen += 1
            if raw.get("is_filtered"):
                self.pre_filter_dropped += 1
                for name, hit in (raw.get("filters") or {}).items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean.
            raw["filters"] = {}
            raw["is_filtered"] = False
            self.batch_buf.append(raw)

    # ── flush signal ──────────────────────────────────────────────────────

    def batch_ready(self) -> bool:
        if self.batch_size is not None:
            return len(self.batch_buf) >= self.batch_size
        assert self.token_batch_size is not None
        return sum(get_seq_len(r) for r in self.batch_buf) >= self.token_batch_size

    def pop(self) -> list[vf.RolloutOutput]:
        """Slice one batch off the front of the buffer. Caller must check ``batch_ready()`` first."""
        if self.batch_size is not None:
            batch = self.batch_buf[: self.batch_size]
            self.batch_buf = self.batch_buf[self.batch_size :]
            return batch
        assert self.token_batch_size is not None
        cut = 0
        running = 0
        for i, r in enumerate(self.batch_buf):
            running += get_seq_len(r)
            cut = i + 1
            if running >= self.token_batch_size:
                break
        batch = self.batch_buf[:cut]
        self.batch_buf = self.batch_buf[cut:]
        return batch

    @property
    def buffered_count(self) -> int:
        return len(self.batch_buf)

    def reset_pre_filter_stats(self) -> None:
        """Clear the per-batch pre-filter counters."""
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
