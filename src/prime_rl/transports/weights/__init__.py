from pathlib import Path

import torch

from prime_rl.configs.shared import WeightBroadcastConfig
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import WeightReceiver, WeightSender, prune_broadcasts_beyond
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver, FileSystemWeightSender
from prime_rl.transports.weights.nccl import NCCLWeightReceiver, NCCLWeightSender
from prime_rl.transports.weights.nixl import NIXLWeightReceiver, NIXLWeightSender

__all__ = [
    "WeightReceiver",
    "WeightSender",
    "prune_broadcasts_beyond",
    "setup_weight_receiver",
    "setup_weight_sender",
]


def setup_weight_sender(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
) -> WeightSender:
    if config.type == "nccl":
        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightSender(output_dir, config)
    elif config.type == "nixl":
        return NIXLWeightSender(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_plane: AdminPlane,
) -> WeightReceiver:
    if config.type == "nccl":
        return NCCLWeightReceiver(broadcast_dir, config, admin_plane)
    elif config.type == "filesystem":
        return FileSystemWeightReceiver(broadcast_dir, config, admin_plane)
    elif config.type == "nixl":
        return NIXLWeightReceiver(broadcast_dir, config, admin_plane)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
