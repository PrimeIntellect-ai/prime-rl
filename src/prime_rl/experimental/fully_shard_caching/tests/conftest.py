import os

import pytest
import torch
import torch.distributed as dist

from prime_rl.experimental.fully_shard_caching.prepared_tensor import PREPARE_CALLS


@pytest.fixture(scope="session")
def single_rank_process_group():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", 0))
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def reset_prepare_calls():
    PREPARE_CALLS.reset()
    yield
