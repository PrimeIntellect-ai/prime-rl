import os
import subprocess

import pytest

from zeroband.utils.world_info import get_world_info, reset_world_info

ENV_VARS = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]


def reset_env_vars():
    for var in ENV_VARS:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture(autouse=True)
def setup():
    reset_env_vars()
    reset_world_info()
    yield


def test_init_with_default_args():
    world_info = get_world_info()

    # Test class attributes
    assert world_info.world_size == world_info.local_world_size == 1
    assert world_info.rank == world_info.local_rank == 0
    assert world_info.num_nodes == 1
    assert world_info == get_world_info()

    # Test JSON output
    world_info_dict = get_world_info().json()
    assert type(world_info_dict) is dict
    assert list(world_info_dict.keys()) == list(map(lambda x: x.lower(), ENV_VARS)) + ["num_nodes"]
    assert list(world_info_dict.values()) == [0, 1, 0, 1, 1]


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
    world_info = get_world_info()
    assert world_info.world_size == world_size
    assert world_info.local_world_size == local_world_size
    assert world_info.rank == world_info.local_rank == 0
    assert world_info.num_nodes == world_size // local_world_size
    assert world_info == get_world_info()


@pytest.mark.parametrize("local_world_size", [1, 2])
def test_init_with_torchrun(local_world_size: int):
    path = "src/zeroband/utils/world_info.py"
    assert os.path.exists(path)
    cmd = ["torchrun", f"--nproc_per_node={local_world_size}", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_msg = stderr.decode("utf-8").strip() or stdout.decode("utf-8").strip() or f"Process failed with code {process.returncode}"
        pytest.fail(f"Process failed: {error_msg}")


def test_init_with_invalid_local_world_size():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    with pytest.raises(AssertionError):
        get_world_info()


@pytest.mark.parametrize("rank_world_size", [(1, 1), (-1, 1)])
def test_init_with_invalid_rank(rank_world_size: tuple[int, int]):
    rank, world_size = rank_world_size
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    with pytest.raises(AssertionError):
        get_world_info()


@pytest.mark.parametrize("local_rank_world_size", [(1, 1), (-1, 1)])
def test_init_with_invalid_local_rank(local_rank_world_size: tuple[int, int]):
    local_rank, world_size = local_rank_world_size
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    with pytest.raises(AssertionError):
        get_world_info()
