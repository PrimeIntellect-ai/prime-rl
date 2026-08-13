import asyncio
from unittest.mock import AsyncMock, MagicMock

from prime_rl.utils.client import init_nccl_broadcast, update_weights


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


def test_native_full_weight_update_uses_positional_path(tmp_path):
    client = AsyncMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post.return_value = response

    asyncio.run(update_weights([client], tmp_path, step=2, use_native_collective_rpc=True))

    collective_calls = [call for call in client.post.await_args_list if call.args[0] == "/collective_rpc"]
    assert collective_calls[0].kwargs["json"] == {
        "method": "update_weights_from_path",
        "args": [tmp_path.as_posix()],
    }
