from pathlib import Path

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import WeightBroadcast
from prime_rl.transports.weights.filesystem import FileSystemWeightBroadcast
from prime_rl.transports.weights.nccl import NCCLWeightBroadcast
from prime_rl.transports.weights.nixl import NIXLWeightBroadcast
from prime_rl.transports.weights.mx_refit import MXRefitWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
    model_name: str | None = None,
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "nixl":
        return NIXLWeightBroadcast(output_dir, config, parallel_dims)
    elif config.type == "mx_refit":
        if model_name is None:
            raise ValueError("mx_refit weight broadcast requires model_name")
        return MXRefitWeightBroadcast(output_dir, config, parallel_dims, model_name)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
