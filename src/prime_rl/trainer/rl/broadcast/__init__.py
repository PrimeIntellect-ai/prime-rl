from pathlib import Path

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: WeightBroadcastConfig,
    lora_config: LoRAConfig | None = None,
    parallel_dims: ParallelDims | None = None,
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "nixl":
        if parallel_dims is None:
            raise ValueError("parallel_dims is required for NIXL weight transfer")
        return NIXLWeightBroadcast(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
