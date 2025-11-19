import time
from pathlib import Path
from typing import Literal

from torch import nn

from prime_rl.trainer.lora import has_lora_layers
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from prime_rl.trainer.weights import convert_tt_to_hf_moe, gather_weights_on_master, has_tt_moe_layers, quantize_layer_params, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(self, output_dir: Path, config: FileSystemWeightBroadcastConfig):
        super().__init__(output_dir)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded
        self.world = get_world()
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={config.save_sharded}, broadcast_dir={self.broadcast_dir})"
        )
        self.use_fp8 = config.use_fp8

    def broadcast_weights(self, model: nn.Module, step: int):
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.time()
        has_lora = has_lora_layers(model)
        state_dict = gather_weights_on_master(model, is_master=self.world.is_master, has_lora_layers=has_lora)
        if self.world.is_master:
            # Convert TT-MoE layers to HF format if needed
            if has_tt_moe_layers(state_dict):
                convert_tt_to_hf_moe(state_dict)
                
            if self.use_fp8:
                quantize_layer_params(state_dict)

            # Save weights to shared filesystem
            save_dir = get_step_path(self.broadcast_dir, step)
            save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded)

            # Notify the orchestrator at the end of step to signal that it is safe to load weights from shared filesystem
            self.notify_orchestrator(step)
            self.logger.debug(f"Weights broadcasted in {time.time() - start_time:.2f}s")
