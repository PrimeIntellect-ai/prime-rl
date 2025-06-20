from typing import Literal, cast

from datasets import Dataset, load_dataset

Benchmark = Literal["math500", "aime24", "aime25"]

_BENCHMARKS_DATASET_NAMES: dict[Benchmark, str] = {
    "math500": "justus27/math-500-genesys",  # "PrimeIntellect/MATH-500",
    "aime24": "justus27/aime-24-genesys",  # "PrimeIntellect/AIME-24",
    "aime25": "justus27/aime-25-genesys",  # "PrimeIntellect/AIME-25",
}

_BENCHMARK_DISPLAY_NAMES: dict[Benchmark, str] = {
    "math500": "MATH-500",
    "aime24": "AIME-24",
    "aime25": "AIME-25",
}


def get_benchmark_dataset(name: Benchmark) -> Dataset:
    return cast(Dataset, load_dataset(_BENCHMARKS_DATASET_NAMES[name], split="train"))


def get_benchmark_display_name(name: Benchmark) -> str:
    return _BENCHMARK_DISPLAY_NAMES[name]
