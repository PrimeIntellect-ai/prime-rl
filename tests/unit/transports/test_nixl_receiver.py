import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, sentinel

from prime_rl.transports.weights.nixl.nixl import NIXLWeightReceiver


def test_nixl_receiver_checks_topology_before_collective_update():
    receiver = NIXLWeightReceiver.__new__(NIXLWeightReceiver)
    receiver.topology_guard = AsyncMock()
    receiver._ack = MagicMock()
    receiver.trainer_peer = sentinel.trainer_peer
    receiver.nixl_agent = MagicMock()
    receiver.config = MagicMock(timeout=1200)
    receiver.admin_clients = []
    receiver.use_collective_rpc = True

    with (
        patch(
            "prime_rl.transports.weights.nixl.nixl.asyncio.to_thread",
            new_callable=AsyncMock,
        ),
        patch(
            "prime_rl.transports.weights.nixl.nixl.update_weights",
            new_callable=AsyncMock,
        ) as update,
    ):
        asyncio.run(receiver.receive(4))

    receiver.topology_guard.assert_awaited_once_with()
    receiver._ack.assert_called_once_with(4)
    update.assert_awaited_once_with(
        [],
        None,
        step=4,
        use_collective_rpc=True,
    )
