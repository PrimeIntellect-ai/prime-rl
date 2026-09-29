from pathlib import Path

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import (
    WeightReceiver,
    WeightSender,
    prune_broadcasts_beyond,
    reclaim_memory_for_broadcast,
)
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver, FileSystemWeightSender
from prime_rl.transports.weights.nccl import NCCLWeightReceiver, NCCLWeightSender
from prime_rl.utils.mx_compat import import_model_express, require_mx_refit

__all__ = [
    "WeightReceiver",
    "WeightSender",
    "prune_broadcasts_beyond",
    "reclaim_memory_for_broadcast",
    "setup_weight_receiver",
    "setup_weight_sender",
]


def _nixl_classes():
    import_model_express("modelexpress.client", transport="nixl")
    from prime_rl.transports.weights.nixl import NIXLWeightReceiver, NIXLWeightSender

    return NIXLWeightSender, NIXLWeightReceiver


def _mx_refit_classes(staging_mode: str | None = None):
    require_mx_refit(staging_mode=staging_mode)
    from prime_rl.transports.weights.mx_refit import MXRefitWeightReceiver, MXRefitWeightSender

    return MXRefitWeightSender, MXRefitWeightReceiver


def setup_weight_sender(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
    model_name: str | None = None,
) -> WeightSender:
    if config.type == "nccl":
        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightSender(output_dir, config, lora_config)
    elif config.type == "nixl":
        sender_cls, _ = _nixl_classes()
        return sender_cls(output_dir, config, parallel_dims)
    elif config.type == "mx_refit":
        if model_name is None:
            raise ValueError("mx_refit weight broadcast requires model_name")
        sender_cls, _ = _mx_refit_classes(staging_mode=config.staging_mode)
        return sender_cls(output_dir, config, parallel_dims, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_plane: AdminPlane,
    model_name: str,
) -> WeightReceiver:
    if config.type == "nccl":
        return NCCLWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "filesystem":
        return FileSystemWeightReceiver(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "nixl":
        _, receiver_cls = _nixl_classes()
        return receiver_cls(broadcast_dir, config, admin_plane, model_name)
    elif config.type == "mx_refit":
        _, receiver_cls = _mx_refit_classes()
        return receiver_cls(broadcast_dir, config, admin_plane, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
