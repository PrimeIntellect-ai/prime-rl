from dataclasses import dataclass


@dataclass(frozen=True)
class SampleDescriptor:
    position: int
    length: int


class OnlinePacker:
    """Online best fit over one optimizer step, stopping at the first unplaceable sample."""

    def __init__(self, capacity: int, num_rows: int):
        if capacity < 1 or num_rows < 1:
            raise ValueError("Packing capacity and row count must be positive")
        self.capacity = capacity
        self.rows: list[list[SampleDescriptor]] = [[] for _ in range(num_rows)]
        self.remaining = [capacity] * num_rows

    @property
    def full(self) -> bool:
        return not any(self.remaining)

    def add(self, sample: SampleDescriptor) -> bool:
        if not 0 < sample.length <= self.capacity:
            raise ValueError(f"Invalid processed sample length: {sample.length}")
        candidates = [index for index, space in enumerate(self.remaining) if space >= sample.length]
        if not candidates:
            return False
        index = min(candidates, key=lambda index: (self.remaining[index], index))
        self.rows[index].append(sample)
        self.remaining[index] -= sample.length
        return True


def schedule_rows(rows: list[list[SampleDescriptor]], dp_size: int) -> list[list[list[SampleDescriptor]]]:
    """Group adjacent attention costs in each microstep across data-parallel ranks."""
    if dp_size < 1 or len(rows) % dp_size:
        raise ValueError("Packed row count must be divisible by the DP size")
    ordered = sorted(rows, key=lambda row: sum(sample.length**2 for sample in row))
    return [ordered[rank::dp_size] for rank in range(dp_size)]
