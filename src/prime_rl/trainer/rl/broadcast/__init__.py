from pathlib import Path

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast


def setup_weight_broadcast(
    output_dir: Path, config: WeightBroadcastConfig, lora_config: LoRAConfig | None = None
) -> WeightBroadcast:
    if config.type == "nccl":
        from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast

        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "async_filesystem":
        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        return AsyncFileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "filesystem":
        from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast

        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
