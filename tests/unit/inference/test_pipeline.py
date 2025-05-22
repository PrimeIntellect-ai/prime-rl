from multiprocessing import Process, Queue

import pytest
from prime_iroh import Node

import zeroband.utils.envs as envs  # noqa
from zeroband.inference.pipeline import setup_comm

# Pre-computed node IDs for different seeds (our team's favorite numbers)
IROH_NODE_ID_MAP = {
    9: "00d21610e478bc59b0c1e70505874e191bf94ab73cb1f9246f963f9bc0a1b253",  # Jack
    10: "ff572e291402ae6a3952e54459c349acd635908e2dd34a7c02f04c88d8a616a6",  # Mika
    11: "f69f4d12b2283bc43a6dd8f0e83df69ffa91cc9e76cca77c0f85b3fa9854f55a",  # Jimmy
    13: "c15efa1d4b0a2f4473c694703df14a70c1da9bca8772db974fd4631c87b90463",  # Manveer
    19: "c45523145ee88ad9322cd0668f64d85a153f42ffb4157584c748bed65ffff85f",  # Sami
    42: "9bdb607f02802cdd126290cfa1e025e4c13bbdbb347a70edeace584159303454",  # Vincent
    99: "affeb073bfca840baab714d67813b21e4671444685217d02b48c10eaa8dcbbb6",  # Johannes
    101: "57bb53127d7ad7c2ea91e87fa57ef18ac6ad6f2ea84c10092ad7693ff2f88a7e",  # Madison
}

SEEDS = list(IROH_NODE_ID_MAP.keys())
TIMEOUT = 30


@pytest.mark.parametrize("seed", SEEDS)
def test_node_seeding(seed):
    node = Node.with_seed(num_streams=1, seed=seed)
    assert node.node_id() == IROH_NODE_ID_MAP[seed]


def _setup_comm(rank: int, world_size: int, error_queue: Queue):
    seed = SEEDS[rank]
    peer_seed = SEEDS[(rank + 1) % world_size]
    peer_id = IROH_NODE_ID_MAP[peer_seed]
    try:
        node = setup_comm(world_size, seed, peer_id)
    except Exception as e:
        error_queue.put((rank, str(e)))
        raise e
    finally:
        node.close()


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.slow
def test_setup_comm(world_size: int):
    # Test that setup_comm raises an error for 1 stage
    if world_size == 1:
        with pytest.raises(AssertionError):
            setup_comm(world_size, None, None)
        return

    # Setup error queue and processes
    error_queue = Queue()
    processes = []
    for rank in range(world_size):
        process = Process(target=_setup_comm, args=(rank, world_size, error_queue))
        processes.append(process)

    # Start processes
    for p in processes:
        p.start()

    # Wait for processes
    for p in processes:
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f"Process took longer than {TIMEOUT} seconds to complete")

    # Check for errors
    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            rank, error = error_queue.get()
            errors.append(f"Rank {rank}: {error}")
        raise RuntimeError("Subprocess errors:\n" + "\n".join(errors))
