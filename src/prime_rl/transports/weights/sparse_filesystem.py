import json
import shutil
import time
from pathlib import Path

import torch.distributed as dist
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig, SparseFileSystemWeightBroadcastConfig
from prime_rl.trainer.sparse_update import commit_sparse_update
from prime_rl.transports.weights.base import WeightSender


class SparseFileSystemWeightSender(WeightSender):
    """Publish rank-local sparse patches captured after the optimizer step."""

    def __init__(
        self,
        output_dir: Path,
        config: SparseFileSystemWeightBroadcastConfig,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        super().__init__(output_dir, config.timeout)
        if lora_config is not None:
            raise ValueError("Sparse filesystem updates do not support LoRA")
        self.staging_dir = output_dir / ".sparse_weight_updates"

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        started = time.perf_counter()
        if dist.is_initialized():
            dist.barrier()
        if self.world.is_master:
            staged_step_dir = self.staging_dir / f"step_{step}"
            manifest_path = commit_sparse_update(
                self.staging_dir,
                target_step=step,
                base_step=max(0, step - 1),
                world_size=self.world.world_size,
            )
            manifest = json.loads(manifest_path.read_text())
            for source in staged_step_dir.iterdir():
                shutil.move(str(source), step_dir / source.name)
            staged_step_dir.rmdir()
            self.logger.info(
                f"Published distributed sparse policy v{step}: {manifest['changed']:,} changed values, "
                f"{manifest['payload_bytes'] / 1024**2:.2f} MiB in {time.perf_counter() - started:.2f}s"
            )
        if dist.is_initialized():
            dist.barrier()
