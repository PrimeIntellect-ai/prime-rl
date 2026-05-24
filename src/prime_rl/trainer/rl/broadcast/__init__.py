from pathlib import Path

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: WeightBroadcastConfig,
    lora_config: LoRAConfig | None = None,
    parallel_dims=None,
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "nixl_mx":
        from prime_rl.trainer.rl.broadcast.nixl_mx import NIXLMxWeightBroadcast

        assert parallel_dims is not None, "nixl_mx requires parallel_dims"
        return NIXLMxWeightBroadcast(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
