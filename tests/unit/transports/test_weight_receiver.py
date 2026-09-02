from unittest.mock import AsyncMock, patch, sentinel

import pytest

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, NIXLWeightBroadcastConfig
from prime_rl.transports.weights import setup_weight_receiver


@pytest.mark.parametrize(
    "config",
    [FileSystemWeightBroadcastConfig(), NIXLWeightBroadcastConfig()],
)
@pytest.mark.parametrize(
    "dynamo_controls",
    [
        {"use_collective_rpc": True},
        {"worker_world_sizes": (1,)},
    ],
)
def test_non_nccl_receiver_rejects_dynamo_controls(tmp_path, config, dynamo_controls):
    with pytest.raises(ValueError, match="Dynamo discovery requires the NCCL weight receiver"):
        setup_weight_receiver(
            tmp_path,
            config,
            [],
            "Qwen/Qwen3-0.6B",
            **dynamo_controls,
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
