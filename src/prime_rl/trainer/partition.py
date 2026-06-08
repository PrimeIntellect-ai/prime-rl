"""Balanced multiway partitioning for distributing micro-batches across workers.

Splits weighted item indices into ``num_partitions`` groups of near-equal total
weight using the Karmarkar-Karp largest-differencing heuristic, then a local-search
swap refinement that minimizes ``(max_load, max_load - min_load)``.
"""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class _WeightedSet:
    total: int = 0
    items: list[int] = field(default_factory=list)

    def add(self, idx: int, weight: int) -> None:
        self.items.append(idx)
        self.total += weight

    def merge(self, other: "_WeightedSet") -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: "_WeightedSet") -> bool:
        if self.total != other.total:
            return self.total < other.total
        return self.items < other.items


class _KKState:
    def __init__(self, items: list[tuple[int, int]], k: int):
        self.sets = [_WeightedSet() for _ in range(k)]
        for set_idx, (idx, weight) in enumerate(items):
            self.sets[set_idx].add(idx, weight)
        self.sets.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].total - self.sets[-1].total

    def merge(self, other: "_KKState") -> None:
        k = len(self.sets)
        for i in range(k):
            self.sets[i].merge(other.sets[k - 1 - i])
        self.sets.sort(reverse=True)

    def partitions(self) -> list[list[int]]:
        return [sorted(weighted_set.items) for weighted_set in self.sets]

    def __lt__(self, other: "_KKState") -> bool:
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]


def _karmarkar_karp(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    assert len(weights) >= num_partitions
    assert len(weights) % num_partitions == 0
    weighted_indices = sorted((weight, idx) for idx, weight in enumerate(weights))
    states: list[_KKState] = []
    for offset in range(0, len(weighted_indices), num_partitions):
        items = [(idx, weight) for weight, idx in weighted_indices[offset : offset + num_partitions]]
        heapq.heappush(states, _KKState(items, num_partitions))

    while len(states) > 1:
        state = heapq.heappop(states)
        state.merge(heapq.heappop(states))
        heapq.heappush(states, state)

    return states[0].partitions()


def _partition_loads(weights: Sequence[int], partitions: list[list[int]]) -> list[int]:
    return [sum(weights[i] for i in partition) for partition in partitions]


def _refine_by_swapping(weights: Sequence[int], partitions: list[list[int]]) -> list[list[int]]:
    partitions = [list(partition) for partition in partitions]
    loads = _partition_loads(weights, partitions)

    while True:
        best_swap = None
        best_score = (max(loads), max(loads) - min(loads))
        for left_rank in range(len(partitions)):
            for right_rank in range(left_rank + 1, len(partitions)):
                for left_pos, left_idx in enumerate(partitions[left_rank]):
                    for right_pos, right_idx in enumerate(partitions[right_rank]):
                        new_left = loads[left_rank] - weights[left_idx] + weights[right_idx]
                        new_right = loads[right_rank] - weights[right_idx] + weights[left_idx]
                        new_loads = list(loads)
                        new_loads[left_rank] = new_left
                        new_loads[right_rank] = new_right
                        score = (max(new_loads), max(new_loads) - min(new_loads))
                        if score < best_score:
                            best_score = score
                            best_swap = (left_rank, right_rank, left_pos, right_pos, new_loads)
        if best_swap is None:
            return partitions

        left_rank, right_rank, left_pos, right_pos, loads = best_swap
        partitions[left_rank][left_pos], partitions[right_rank][right_pos] = (
            partitions[right_rank][right_pos],
            partitions[left_rank][left_pos],
        )


def balanced_partition(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    """Partition item indices into ``num_partitions`` groups of near-equal total weight.

    Requires ``len(weights)`` to be a positive multiple of ``num_partitions``.
    """
    partitions = _karmarkar_karp(weights, num_partitions)
    return _refine_by_swapping(weights, partitions)
