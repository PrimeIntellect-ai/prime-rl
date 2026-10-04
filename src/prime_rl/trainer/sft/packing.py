from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterator


@dataclass(frozen=True)
class SampleDescriptor:
    sample_id: tuple[int, int]
    length: int
    compatibility: Hashable = "text"
    num_bytes: int = 0
    arrival_step: int = 0


class OnlinePacker:
    """Deterministic best fit over a fixed global optimizer-step capacity."""

    def __init__(
        self,
        source: Iterator[SampleDescriptor],
        capacity: int,
        num_rows: int,
        dp_size: int = 1,
        lookahead_samples: int = 0,
        max_pending_samples: int = 128,
        max_pending_bytes: int = 256 * 1024**2,
        max_sample_bytes: int = 64 * 1024**2,
    ):
        if capacity < 1 or num_rows < 1 or dp_size < 1 or num_rows % dp_size:
            raise ValueError("Positive capacity and a whole number of DP rounds are required")
        if not 0 <= lookahead_samples < max_pending_samples:
            raise ValueError("lookahead_samples must be nonnegative and smaller than max_pending_samples")
        if not 0 < max_sample_bytes <= max_pending_bytes:
            raise ValueError("max_pending_bytes must hold at least one max_sample_bytes sample")
        self.source = source
        self.capacity = capacity
        self.num_rows = num_rows
        self.dp_size = dp_size
        self.lookahead_samples = lookahead_samples
        self.max_pending_samples = max_pending_samples
        self.max_pending_bytes = max_pending_bytes
        self.max_sample_bytes = max_sample_bytes
        self.pending: deque[SampleDescriptor] = deque()
        self.step = 0
        self.exhausted = False
        self.metrics: dict[str, int] = {}

    def state_dict(self) -> dict:
        return {"pending": list(self.pending), "step": self.step, "exhausted": self.exhausted}

    def load_state_dict(self, state: dict) -> None:
        self.pending = deque(state["pending"])
        self.step = state["step"]
        self.exhausted = state["exhausted"]

    def next_step(self) -> list[list[SampleDescriptor]] | None:
        buckets: list[list[SampleDescriptor]] = [[] for _ in range(self.num_rows)]
        free = [self.capacity] * self.num_rows
        round_keys: dict[int, Hashable] = {}
        held: deque[SampleDescriptor] = deque()
        held_bytes = 0
        blocked = False
        inspected = 0
        moved = 0
        max_wait = 0
        while any(free):
            if blocked:
                if inspected >= self.lookahead_samples:
                    break
                if not self.pending and (
                    len(held) >= self.max_pending_samples or held_bytes + self.max_sample_bytes > self.max_pending_bytes
                ):
                    break
            if self.pending:
                sample = self.pending.popleft()
            elif self.exhausted:
                break
            else:
                sample = next(self.source, None)
                if sample is None:
                    self.exhausted = True
                    break
            if not 0 < sample.length <= self.capacity:
                raise ValueError(f"Sample {sample.sample_id} has unresolved length {sample.length}")
            if not 0 <= sample.num_bytes <= self.max_sample_bytes:
                raise ValueError(f"Sample {sample.sample_id} exceeds max_sample_bytes")
            if blocked:
                inspected += 1

            candidates = [
                index
                for index, bucket in enumerate(buckets)
                if bucket and bucket[0].compatibility == sample.compatibility and free[index] >= sample.length
            ]
            if candidates:
                index = min(candidates, key=lambda index: (free[index], index))
            else:
                index = next(
                    (
                        index
                        for index, bucket in enumerate(buckets)
                        if not bucket
                        and round_keys.get(index // self.dp_size, sample.compatibility) == sample.compatibility
                    ),
                    None,
                )
            if index is None:
                blocked = True
                held.append(sample)
                held_bytes += sample.num_bytes
                continue
            buckets[index].append(sample)
            free[index] -= sample.length
            round_keys[index // self.dp_size] = sample.compatibility
            moved += int(blocked)
            max_wait = max(max_wait, self.step - sample.arrival_step)

        held.extend(self.pending)
        self.pending = held
        pending_bytes = sum(sample.num_bytes for sample in held)
        if len(held) > self.max_pending_samples or pending_bytes > self.max_pending_bytes:
            raise ValueError("Pending samples exceed the configured buffer limits")
        self.metrics = {
            "pending_samples": len(held),
            "pending_bytes": pending_bytes,
            "max_wait_steps": max_wait,
            "skipped_ahead_samples": moved,
        }
        if not any(buckets):
            return None
        self.step += 1
        return buckets


def schedule_rows(buckets: list[list[SampleDescriptor]], dp_size: int) -> list[list[list[SampleDescriptor]]]:
    """Balance segment attention work while preserving modality-aligned rounds."""
    lanes: list[list[list[SampleDescriptor]]] = [[] for _ in range(dp_size)]
    costs = [0] * dp_size
    for offset in range(0, len(buckets), dp_size):
        rows = buckets[offset : offset + dp_size]
        row_costs = [sum(sample.length**2 for sample in row) for row in rows]
        lane_order = sorted(range(dp_size), key=lambda lane: (costs[lane], lane))
        row_order = sorted(range(dp_size), key=lambda row: (-row_costs[row], row))
        for lane, row in zip(lane_order, row_order, strict=True):
            lanes[lane].append(rows[row])
            costs[lane] += row_costs[row]
    return lanes
