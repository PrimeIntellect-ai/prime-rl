import pytest
from prime_iroh import Node
from multiprocessing import Process

from zeroband.inference.pipeline import setup_comm

# Pre-computed node IDs for different seeds (useful for debugging)
IROH_NODE_ID_MAP = {
    0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    2: "191fc38f134aaf1b7fdb1f86330b9d03e94bd4ba884f490389de964448e89b3f",
    3: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
    4: "4f44e6c7bdfed3d9f48d86149ee3d29382cae8c83ca253e06a70be54a301828b",
    5: "e2e8aa145e1ec5cb01ebfaa40e10e12f0230c832fd8135470c001cb86d77de00",
    6: "17888c2ca502371245e5e35d5bcf35246c3bc36878e859938c9ead3c54db174f",
    7: "478243aed376da313d7cf3a60637c264cb36acc936efb341ff8d3d712092d244",
}

TIMEOUT = 10
SEEDS = list(range(8))


@pytest.mark.parametrize("seed", SEEDS)
def test_has_precomputed_node_ids(seed):
    assert len(SEEDS) == len(IROH_NODE_ID_MAP)
    assert IROH_NODE_ID_MAP.get(seed) is not None


@pytest.mark.parametrize("seed", range(8))
def test_node_seeding(seed):
    node = Node.with_seed(num_streams=1, seed=seed)
    assert node.node_id() == IROH_NODE_ID_MAP[seed]


def _setup_comm(rank: int, world_size: int):
    peer_id = IROH_NODE_ID_MAP[(rank + 1) % world_size]
    node = setup_comm(world_size, rank, peer_id)
    assert isinstance(node, Node)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_setup_comm(world_size: int):
    # Test that setup_comm raises an error for 1 stage
    if world_size == 1:
        with pytest.raises(AssertionError):
            setup_comm(world_size, None, None)

    # Setup processes
    processes = []
    for rank in range(world_size):
        process = Process(target=_setup_comm, args=(rank, world_size))
        processes.append(process)

    # Start processes
    for p in processes:
        p.start()

    # Terminate processes (raise exception with timeout)
    for p in processes:
        try:
            p.join(timeout=TIMEOUT)
        except TimeoutError:
            p.terminate()
            raise TimeoutError(f"Process took longer than {TIMEOUT} seconds to complete")
