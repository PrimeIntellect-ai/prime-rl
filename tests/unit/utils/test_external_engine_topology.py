import asyncio
from unittest.mock import AsyncMock, MagicMock

from prime_rl.utils.client import _rank_offsets, init_nccl_broadcast


def test_rank_offsets_support_heterogeneous_engines():
    assert _rank_offsets([1, 3, 2], inference_world_size=6) == [0, 1, 4]


def test_nccl_broadcast_forwards_explicit_engine_sizes():
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
            engine_world_sizes=[1, 3],
        )
    )

    assert [call.kwargs["json"]["rank_offset"] for client in clients for call in client.post.await_args_list] == [0, 1]
    assert [call.kwargs["json"]["engine_world_size"] for client in clients for call in client.post.await_args_list] == [
        1,
        3,
    ]


def test_nccl_broadcast_preserves_legacy_payload():
    client = AsyncMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post.return_value = response

    asyncio.run(init_nccl_broadcast([client], "127.0.0.1", 29519, 1200, 1))

    assert "engine_world_size" not in client.post.await_args.kwargs["json"]
