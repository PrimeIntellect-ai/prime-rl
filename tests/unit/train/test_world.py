import os

import pytest

from prime_rl.trainer.world import get_world

ENV_VARS = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]


def test_init_with_default_args():
    world = get_world()

    # Test class attributes
    assert world.world_size == world.local_world_size == 1
    assert world.rank == world.local_rank == 0
    assert world.num_nodes == 1
    assert world == get_world()


@pytest.mark.parametrize("local_world_size", [1, 2])
@pytest.mark.parametrize("world_size", [1, 2])
def test_init_with_valid_env_vars(local_world_size: int, world_size: int):
    # Invalid env vars, skip test
    if local_world_size > world_size:
        return
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    world = get_world()
    assert world.world_size == world_size
    assert world.local_world_size == local_world_size
    assert world.rank == world.local_rank == 0
    assert world.num_nodes == world_size // local_world_size
    assert world == get_world()


def test_init_with_invalid_local_world_size():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    with pytest.raises(AssertionError):
        get_world()


@pytest.mark.parametrize("rank_world_size", [(1, 1), (-1, 1)])
def test_init_with_invalid_rank(rank_world_size: tuple[int, int]):
    rank, world_size = rank_world_size
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    with pytest.raises(AssertionError):
        get_world()


@pytest.mark.parametrize("local_rank_world_size", [(1, 1), (-1, 1)])
def test_init_with_invalid_local_rank(local_rank_world_size: tuple[int, int]):
    local_rank, world_size = local_rank_world_size
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    with pytest.raises(AssertionError):
        get_world()


def test_default_torchrun_env_defaults_single_process(monkeypatch):
    """Outside torchrun, the rendezvous env defaults to a single-process group."""
    from prime_rl.trainer.utils import default_torchrun_env

    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    default_torchrun_env()
    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert int(os.environ["MASTER_PORT"]) > 0

    # The defaulted env satisfies the env:// rendezvous (CPU backend).
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    assert dist.get_world_size() == 1
    assert dist.get_rank() == 0
    dist.destroy_process_group()


def test_default_torchrun_env_noop_under_torchrun(monkeypatch):
    """Under torchrun the env is already set and stays untouched."""
    from prime_rl.trainer.utils import default_torchrun_env

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    default_torchrun_env()
    assert os.environ["RANK"] == "3"
    assert os.environ["MASTER_ADDR"] == "10.0.0.1"
