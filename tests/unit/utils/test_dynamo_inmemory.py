import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.utils.client import init_nccl_broadcast
from prime_rl.utils.dynamo import DynamoInferencePool


def test_native_nccl_initialization_uses_collective_rpc():
    clients = [AsyncMock(), AsyncMock()]
    for client in clients:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        client.post.return_value = response

    asyncio.run(
        init_nccl_broadcast(
            clients,
            host="127.0.0.1",
            port=29519,
            timeout=1200,
            inference_world_size=4,
            engine_world_sizes=[2, 2],
            use_native_collective_rpc=True,
        )
    )

    assert [client.post.await_args.args[0] for client in clients] == ["/collective_rpc", "/collective_rpc"]
    assert [client.post.await_args.kwargs["json"]["kwargs"]["rank_offset"] for client in clients] == [0, 2]


def test_dynamo_pool_passes_discovered_topology_to_nccl():
    pool = DynamoInferencePool.__new__(DynamoInferencePool)
    pool._admin_clients = [AsyncMock(), AsyncMock()]
    pool._admin_world_sizes = [1, 3]

    with patch("prime_rl.utils.dynamo.init_nccl_broadcast", new=AsyncMock()) as initialize:
        asyncio.run(
            pool.init_nccl_broadcast(
                host="trainer",
                port=29501,
                timeout=1200,
                inference_world_size=4,
                quantize_in_weight_transfer=False,
            )
        )

    assert initialize.await_args.kwargs["engine_world_sizes"] == [1, 3]
    assert initialize.await_args.kwargs["use_native_collective_rpc"] is True


def test_dynamo_pool_uses_native_full_weight_update():
    pool = DynamoInferencePool.__new__(DynamoInferencePool)
    pool._admin_clients = [AsyncMock()]

    with patch("prime_rl.utils.dynamo.update_weights", new=AsyncMock()) as update:
        asyncio.run(pool.update_weights(Path("/weights"), step=2))

    assert update.await_args.kwargs["use_native_collective_rpc"] is True
