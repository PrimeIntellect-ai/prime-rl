from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.transports.weights.base import WeightReceiver, WeightSender, prune_broadcasts_beyond

if TYPE_CHECKING:
    from prime_rl.trainer.parallel_dims import ParallelDims

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
    lora_config: LoRAConfig | None = None,
) -> WeightSender:
    if config.type == "nccl":
        from prime_rl.transports.weights.nccl import NCCLWeightSender

        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        from prime_rl.transports.weights.filesystem import FileSystemWeightSender

        return FileSystemWeightSender(output_dir, config, lora_config)
    elif config.type == "nixl":
        from prime_rl.transports.weights.nixl import NIXLWeightSender

        return NIXLWeightSender(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_plane: AdminPlane,
    model_name: str,
) -> WeightReceiver:
    if config.type == "nccl":
        from prime_rl.transports.weights.nccl import NCCLWeightReceiver

        return NCCLWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "filesystem":
        from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver

        return FileSystemWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "nixl":
        from prime_rl.transports.weights.nixl import NIXLWeightReceiver

        return NIXLWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
