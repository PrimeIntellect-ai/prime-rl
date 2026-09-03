from unittest.mock import AsyncMock, patch, sentinel

import pytest

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, NIXLWeightBroadcastConfig
from prime_rl.transports.weights import setup_weight_receiver


@pytest.mark.parametrize(
    ("config", "receiver_name"),
    [
        (NIXLWeightBroadcastConfig(), "NIXLWeightReceiver"),
    ],
)
def test_non_nccl_receiver_ignores_nccl_specific_dynamo_controls(tmp_path, config, receiver_name):
    with patch(f"prime_rl.transports.weights.{receiver_name}", return_value=sentinel.receiver):
        receiver = setup_weight_receiver(
            tmp_path,
            config,
            [],
            "Qwen/Qwen3-0.6B",
            worker_world_sizes=(1,),
            use_collective_rpc=True,
            topology_guard=AsyncMock(),
        )

    assert receiver is sentinel.receiver


def test_filesystem_receiver_uses_dynamo_collective_rpc_controls(tmp_path):
    topology_guard = AsyncMock()
    with patch(
        "prime_rl.transports.weights.FileSystemWeightReceiver",
        return_value=sentinel.receiver,
    ) as receiver_cls:
        receiver = setup_weight_receiver(
            tmp_path,
            FileSystemWeightBroadcastConfig(),
            [],
            "Qwen/Qwen3-0.6B",
            use_collective_rpc=True,
            topology_guard=topology_guard,
        )

    assert receiver is sentinel.receiver
    receiver_cls.assert_called_once_with(
        tmp_path,
        FileSystemWeightBroadcastConfig(),
        [],
        "Qwen/Qwen3-0.6B",
        use_collective_rpc=True,
        topology_guard=topology_guard,
    )


@pytest.mark.parametrize(
    ("config", "receiver_name"),
    [
        (FileSystemWeightBroadcastConfig(), "FileSystemWeightReceiver"),
        (NIXLWeightBroadcastConfig(), "NIXLWeightReceiver"),
    ],
)
def test_non_nccl_receiver_accepts_static_topology_guard(tmp_path, config, receiver_name):
    with patch(f"prime_rl.transports.weights.{receiver_name}", return_value=sentinel.receiver):
        receiver = setup_weight_receiver(
            tmp_path,
            config,
            [],
            "Qwen/Qwen3-0.6B",
            topology_guard=AsyncMock(),
        )

    assert receiver is sentinel.receiver
