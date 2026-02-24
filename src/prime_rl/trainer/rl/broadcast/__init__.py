from pathlib import Path

import torch

from prime_rl.configs.rl_trainer import WeightBroadcastConfigType
from prime_rl.configs.shared import LoRAConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path, config: WeightBroadcastConfigType, lora_config: LoRAConfig | None = None
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
