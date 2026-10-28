from functools import cache
from pathlib import Path
from typing import Callable

import pytest
import torch
from huggingface_hub import snapshot_download

from tests.regression.distributed.sft.archs import CASES, write_tiny_checkpoint
from tests.regression.distributed.sft.cases import NUM_GPUS, Arch, assign_shards

TOKENIZER_FILES = ["*.json", "*.txt", "tokenizer*", "*.jinja"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--shard", metavar="I/N", help="run shard I of N (1-based) of the distributed suite")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", f"distributed: needs {NUM_GPUS} GPUs on one node")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep only the tests in this run's `--shard`, and skip distributed tests on a node with too few GPUs."""
    if shard_option := config.getoption("--shard"):
        keep_only_shard(config, items, shard_option)
    skip_distributed_without_enough_gpus(items)


def keep_only_shard(config: pytest.Config, items: list[pytest.Item], shard_option: str) -> None:
    shard, num_shards = parse_shard_option(shard_option)
    shard_of_unit = assign_shards(CASES, num_shards)
    deselected = [item for item in items if shard_of(item, shard_of_unit) != shard]
    items[:] = [item for item in items if shard_of(item, shard_of_unit) == shard]
    config.hook.pytest_deselected(items=deselected)


def parse_shard_option(shard_option: str) -> tuple[int, int]:
    """Parse `I/N`, with I counted from 1, into a 0-based shard index and the shard count."""
    index, num_shards = map(int, shard_option.split("/"))
    if not 1 <= index <= num_shards:
        raise pytest.UsageError(f"--shard {shard_option}: need 1 <= I <= N")
    return index - 1, num_shards


def shard_of(item: pytest.Item, shard_of_unit: dict[tuple[str, bool], int]) -> int:
    """The shard of the item's case unit, or the first shard for a test outside the matrix."""
    case = item.callspec.params.get("case") if hasattr(item, "callspec") else None
    return shard_of_unit[case.shard_unit] if case is not None else 0


def skip_distributed_without_enough_gpus(items: list[pytest.Item]) -> None:
    num_gpus = torch.cuda.device_count()
    if num_gpus >= NUM_GPUS:
        return
    skip = pytest.mark.skip(reason=f"needs {NUM_GPUS} GPUs, found {num_gpus}")
    for item in items:
        if item.get_closest_marker("distributed"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Arch], Path]:
    """Build each arch's seeded tiny checkpoint once per session, on first use."""

    @cache
    def _tiny_checkpoint(arch: Arch) -> Path:
        return write_tiny_checkpoint(arch, tmp_path_factory.mktemp(arch.name))

    return _tiny_checkpoint


@pytest.fixture(scope="session")
def tokenizer_files() -> Callable[[Arch], Path]:
    """Download each arch's tokenizer files once per session, so launches can run with `HF_HUB_OFFLINE=1`."""

    @cache
    def _tokenizer_files(arch: Arch) -> Path:
        return Path(snapshot_download(arch.tokenizer, allow_patterns=TOKENIZER_FILES))

    return _tokenizer_files
