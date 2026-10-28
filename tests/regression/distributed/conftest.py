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


def in_shard(item: pytest.Item, shard: int, shards: dict[tuple[str, bool], int]) -> bool:
    case = item.callspec.params.get("case") if hasattr(item, "callspec") else None
    return shards[case.shard_unit] == shard if case is not None else shard == 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shard_option := config.getoption("--shard"):
        index, num_shards = map(int, shard_option.split("/"))
        if not 1 <= index <= num_shards:
            raise pytest.UsageError(f"--shard {shard_option}: need 1 <= I <= N")
        shards = assign_shards(CASES, num_shards)
        deselected = [item for item in items if not in_shard(item, index - 1, shards)]
        items[:] = [item for item in items if in_shard(item, index - 1, shards)]
        config.hook.pytest_deselected(items=deselected)
    if torch.cuda.device_count() >= NUM_GPUS:
        return
    skip = pytest.mark.skip(reason=f"needs {NUM_GPUS} GPUs, found {torch.cuda.device_count()}")
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
