from unittest.mock import AsyncMock

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
        {"topology_guard": AsyncMock()},
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
