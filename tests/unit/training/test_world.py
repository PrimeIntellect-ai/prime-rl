import os
import subprocess

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


@pytest.mark.parametrize("local_world_size", [1, 2])
def test_init_with_torchrun(local_world_size: int):
    path = "src/prime_rl/trainer/world.py"
    assert os.path.exists(path)
    cmd = ["torchrun", f"--nproc_per_node={local_world_size}", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_msg = (
            stderr.decode("utf-8").strip()
            or stdout.decode("utf-8").strip()
            or f"Process failed with code {process.returncode}"
        )
        pytest.fail(f"Process failed: {error_msg}")


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
