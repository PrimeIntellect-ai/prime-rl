from functools import cache
from pathlib import Path
from typing import Callable

import pytest
import torch

from tests.regression.distributed.tiny_models import write_tiny_checkpoint

NUM_GPUS = 8


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", f"distributed: needs {NUM_GPUS} GPUs on one node")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.device_count() >= NUM_GPUS:
        return
    skip = pytest.mark.skip(reason=f"needs {NUM_GPUS} GPUs, found {torch.cuda.device_count()}")
    for item in items:
        if item.get_closest_marker("distributed"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """Build each arch's seeded tiny checkpoint once per session, on first use."""

    @cache
    def _tiny_checkpoint(arch: str) -> Path:
        return write_tiny_checkpoint(arch, tmp_path_factory.mktemp(arch))

    return _tiny_checkpoint
