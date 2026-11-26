from pathlib import Path

import torch

from prime_rl.configs.shared import WeightBroadcastConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.transports.weights.base import (
    WeightReceiver,
    WeightSender,
    latest_finished_broadcast,
    prune_broadcasts_beyond,
)
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver, FileSystemWeightSender

__all__ = [
    "WeightReceiver",
    "WeightSender",
    "latest_finished_broadcast",
    "prune_broadcasts_beyond",
    "setup_weight_receiver",
    "setup_weight_sender",
]


def setup_weight_sender(
    output_dir: Path,
    config: WeightBroadcastConfig,
    lora_config: LoRAConfig | None = None,
    *,
    model_name: str | None = None,
) -> WeightSender:
    if config.type == "nccl":
        # Deferred: the NCCL transport needs vLLM (the `inference` extra), and this module is
        # imported by every trainer run, not just the ones that broadcast weights.
        from prime_rl.transports.weights.nccl import NCCLWeightSender

        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightSender(output_dir, config, lora_config)
    elif config.type == "modelexpress":
        from prime_rl.transports.weights.modelexpress import ModelExpressWeightSender

        if model_name is None:
            raise ValueError("modelexpress requires model_name")
        return ModelExpressWeightSender(output_dir, config, model_name=model_name)
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
        return FileSystemWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "modelexpress":
        from prime_rl.transports.weights.modelexpress import ModelExpressWeightReceiver

        return ModelExpressWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
