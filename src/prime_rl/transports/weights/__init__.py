from collections.abc import Awaitable, Callable
from pathlib import Path

import torch
from httpx import AsyncClient

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
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
    lora_config: LoRAConfig | None = None,
) -> WeightSender:
    if config.type == "nccl":
        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightSender(output_dir, config, lora_config)
    elif config.type == "nixl":
        return NIXLWeightSender(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_clients: list[AsyncClient],
    model_name: str,
    *,
    worker_world_sizes: tuple[int, ...] | None = None,
    use_collective_rpc: bool = False,
    topology_guard: Callable[[], Awaitable[None]] | None = None,
) -> WeightReceiver:
    if config.type != "nccl" and (
        worker_world_sizes is not None or use_collective_rpc or topology_guard is not None
    ):
        raise ValueError("Dynamo discovery requires the NCCL weight receiver")
    if config.type == "nccl":
        return NCCLWeightReceiver(
            broadcast_dir,
            config,
            admin_clients,
            model_name,
            worker_world_sizes=worker_world_sizes,
            use_collective_rpc=use_collective_rpc,
            topology_guard=topology_guard,
        )
    elif config.type == "filesystem":
        return FileSystemWeightReceiver(broadcast_dir, config, admin_clients, model_name)
    elif config.type == "nixl":
        return NIXLWeightReceiver(broadcast_dir, config, admin_clients, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
