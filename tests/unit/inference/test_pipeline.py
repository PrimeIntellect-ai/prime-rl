from multiprocessing import Process, Queue

import pytest
import torch
from prime_iroh import Node

import zeroband.utils.envs as envs  # noqa
from zeroband.inference.pipeline import PipelineConfig, all_reduce, setup_comm

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
TIMEOUT = 60


@pytest.mark.parametrize("seed", SEEDS)
def test_node_seeding(seed):
    node = Node.with_seed(num_streams=1, seed=seed)
    assert node.node_id() == IROH_NODE_ID_MAP[seed]


def _test_setup_comm(rank: int, world_size: int, error_queue: Queue):
    seed = SEEDS[rank]
    peer_seed = SEEDS[(rank + 1) % world_size]
    peer_id = IROH_NODE_ID_MAP[peer_seed]
    config = PipelineConfig(rank=rank, world_size=world_size, iroh_seed=seed, iroh_peer_id=peer_id)
    try:
        node = setup_comm(config)
    except Exception as e:
        error_queue.put((rank, str(e)))
        raise e
    finally:
        node.close()


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.slow
def test_setup_comm(world_size: int):
    # Test that setup_comm raises an error for 1 stage
    if world_size == 1:
        assert setup_comm(PipelineConfig(world_size=world_size)) is None
        return

    # Setup error queue and processes
    error_queue = Queue()
    processes = []
    for rank in range(world_size):
        process = Process(target=_test_setup_comm, args=(rank, world_size, error_queue))
        processes.append(process)

    # Start processes
    for p in processes:
        p.start()

    # Wait for processes
    timed_out = False
    for p in processes:
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            timed_out = True
            p.kill()

    if timed_out:
        raise TimeoutError(f"Process took longer than {TIMEOUT} seconds to complete")

    # Check for errors
    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            rank, error = error_queue.get()
            errors.append(f"Rank {rank}: {error}")
        raise RuntimeError("Subprocess errors:\n" + "\n".join(errors))


def test_all_reduce_single_node():
    """Test all_reduce with single node (world_size=1)"""
    config = PipelineConfig(rank=0, world_size=1)
    test_tensor = torch.tensor(42.0)

    # Create a dummy node (won't be used for communication)
    node = Node(num_streams=1)
    try:
        result = all_reduce(node, test_tensor, config, torch.add)
        assert result.item() == 42.0, f"Expected 42.0, got {result.item()}"
    finally:
        node.close()


def _test_all_reduce(rank: int, world_size: int, operation: str, error_queue: Queue, result_queue: Queue):
    """Helper function to test all_reduce in a subprocess"""
    seed = SEEDS[rank]
    peer_seed = SEEDS[(rank + 1) % world_size]
    peer_id = IROH_NODE_ID_MAP[peer_seed]

    try:
        # Create config
        config = PipelineConfig(rank=rank, world_size=world_size, iroh_seed=seed, iroh_peer_id=peer_id)

        # Setup communication
        node = setup_comm(config)

        # Create test tensor - each rank contributes rank + 1
        test_tensor = torch.tensor(float(rank + 1))

        # Choose operation
        if operation == "sum":
            op = torch.add
        elif operation == "min":
            op = torch.min
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Perform all_reduce
        result = all_reduce(node, test_tensor, config, op)

        # Store result for verification
        result_queue.put((rank, result.item()))

    except Exception as e:
        error_queue.put((rank, str(e)))
        raise e
    finally:
        if "node" in locals():
            node.close()


@pytest.mark.parametrize("world_size,operation", [(2, "sum"), (2, "min"), (4, "sum"), (4, "min")])
@pytest.mark.slow
def test_all_reduce(world_size: int, operation: str):
    """Test all_reduce function with different world sizes and operations"""
    # Setup error and result queues
    error_queue = Queue()
    result_queue = Queue()
    processes = []

    # Start processes
    for rank in range(world_size):
        process = Process(target=_test_all_reduce, args=(rank, world_size, operation, error_queue, result_queue))
        processes.append(process)
        process.start()

    # Wait for processes
    timed_out = False
    for p in processes:
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            timed_out = True
            p.kill()

    if timed_out:
        raise TimeoutError(f"Process took longer than {TIMEOUT} seconds to complete")

    # Check for errors
    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            rank, error = error_queue.get()
            errors.append(f"Rank {rank}: {error}")
        raise RuntimeError("Subprocess errors:\n" + "\n".join(errors))

    # Collect and verify results
    results = {}
    while not result_queue.empty():
        rank, result = result_queue.get()
        results[rank] = result

    # Verify all ranks have results
    assert len(results) == world_size, f"Expected {world_size} results, got {len(results)}"

    # Calculate expected result
    input_values = [rank + 1 for rank in range(world_size)]  # Each rank contributes rank + 1
    if operation == "sum":
        expected = sum(input_values)
    elif operation == "min":
        expected = min(input_values)

    # Verify all ranks got the same correct result
    for rank in range(world_size):
        assert rank in results, f"Missing result for rank {rank}"
        assert abs(results[rank] - expected) < 1e-6, f"Rank {rank}: expected {expected}, got {results[rank]}"
