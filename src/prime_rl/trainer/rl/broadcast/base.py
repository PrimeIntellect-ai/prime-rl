from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn

from prime_rl.trainer.config import LoRAConfig
from prime_rl.utils.logger import get_logger


class WeightBroadcast(ABC):
    def __init__(self, output_dir: Path, lora_config: LoRAConfig | None = None):
        self.logger = get_logger()
        self.output_dir = output_dir
        self.lora_config = lora_config

    @abstractmethod
    def broadcast_weights(self, model: nn.Module, step: int):
        pass
