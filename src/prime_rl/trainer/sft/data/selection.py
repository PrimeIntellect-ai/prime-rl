import copy
from typing import Literal

import numpy as np
from datasets import Dataset

from prime_rl.configs.sft import deterministic_source_counts


class DeterministicMixture:
    """Random-access source schedule shared by every rank and worker.

    Each full cycle has exact quotas. Source rows advance without replacement
    until exhaustion; all_exhausted then wraps them, like HF interleaving.
    The final cycle can be partial. Epoch shuffling changes source row order
    while preserving the schedule and its exhaustion boundary.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        probabilities: list[float] | None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"],
        seed: int,
    ):
        counts = deterministic_source_counts(probabilities, len(datasets))
        self.datasets = [dataset for dataset, count in zip(datasets, counts) if count]
        self.counts = np.array([count for count in counts if count], dtype=np.int64)
        if not self.datasets or any(not len(dataset) for dataset in self.datasets):
            raise ValueError("Deterministic sampling requires nonempty sources with positive probability")
        self.seed = seed
        self.cycle_size = int(self.counts.sum())
        self.column_names = list(dict.fromkeys(column for dataset in self.datasets for column in dataset.column_names))
        self._cycle = None
        self._sources = self._offsets = None

        last_cycles = [(len(dataset) - 1) // int(count) for dataset, count in zip(self.datasets, self.counts)]
        boundary = min(last_cycles) if stopping_strategy == "first_exhausted" else max(last_cycles)
        self._load_cycle(boundary)
        ends = []
        for source, last_cycle in enumerate(last_cycles):
            if last_cycle == boundary:
                remaining = len(self.datasets[source]) - boundary * int(self.counts[source])
                ends.append(int(np.flatnonzero(self._sources == source)[remaining - 1]) + 1)
        end = min(ends) if stopping_strategy == "first_exhausted" else max(ends)
        self.size = boundary * self.cycle_size + end

    def _load_cycle(self, cycle: int):
        if cycle == self._cycle:
            return
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, cycle]))
        self._sources = generator.permutation(np.repeat(np.arange(len(self.datasets)), self.counts))
        self._offsets = np.empty(self.cycle_size, dtype=np.int64)
        for source, count in enumerate(self.counts):
            self._offsets[self._sources == source] = np.arange(count)
        self._cycle = cycle

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.size:
            raise IndexError(index)
        cycle, offset = divmod(index, self.cycle_size)
        self._load_cycle(cycle)
        source = int(self._sources[offset])
        row = (cycle * int(self.counts[source]) + int(self._offsets[offset])) % len(self.datasets[source])
        return self.datasets[source][row]

    def shuffle(self, seed: int, keep_in_memory: bool = False):
        shuffled = copy.copy(self)
        shuffled.datasets = [dataset.shuffle(seed=seed, keep_in_memory=keep_in_memory) for dataset in self.datasets]
        return shuffled

    def take(self, count: int):
        if count < 0:
            raise ValueError("Sample count must be nonnegative")
        selected = copy.copy(self)
        selected.size = min(count, self.size)
        cycle, offset = divmod(selected.size, self.cycle_size)
        self._load_cycle(cycle)
        draws = cycle * self.counts + np.bincount(self._sources[:offset], minlength=len(self.datasets))
        selected.datasets = [dataset.take(min(int(draws[i]), len(dataset))) for i, dataset in enumerate(self.datasets)]
        return selected

    def unique(self, column: str) -> list:
        return list(
            dict.fromkeys(
                value for dataset in self.datasets if column in dataset.column_names for value in dataset.unique(column)
            )
        )
