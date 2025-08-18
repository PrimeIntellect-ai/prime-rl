from typing import Literal

from datasets import Dataset, load_dataset

Benchmark = Literal["math500", "aime24", "aime25", "livecodebench-v5"]

_BENCHMARKS_DATASET_NAMES: dict[Benchmark, str] = {
    "math500": "PrimeIntellect/MATH-500",
    "aime24": "PrimeIntellect/AIME-24",
    "aime25": "PrimeIntellect/AIME-25",
    "livecodebench-v5": "PrimeIntellect/LiveCodeBench-v5",
}

_BENCHMARK_DISPLAY_NAMES: dict[Benchmark, str] = {
    "math500": "MATH-500",
    "aime24": "AIME-24",
    "aime25": "AIME-25",
    "livecodebench-v5": "LiveCodeBench-V5",
}


def get_benchmark_dataset(name: Benchmark) -> Dataset:
    dataset = load_dataset(_BENCHMARKS_DATASET_NAMES[name], split="train")
    assert isinstance(dataset, Dataset)
    return dataset


def get_benchmark_display_name(name: Benchmark) -> str:
    return _BENCHMARK_DISPLAY_NAMES[name]
