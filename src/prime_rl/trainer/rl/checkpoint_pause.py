from __future__ import annotations

from pathlib import Path

import torch.distributed as dist

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.world import World
from prime_rl.utils.checkpoint_pause import (
    PAUSED,
    get_pause_step_dir,
    wait_for_marker,
    write_pause_release,
    write_pause_request,
)
from prime_rl.utils.logger import get_logger


class InferenceCheckpointPause:
    def __init__(self, output_dir: Path, config: CheckpointConfig | None, world: World) -> None:
        self.output_dir = output_dir
        self.world = world
        self.enabled = bool(config and config.experimental and config.experimental.pause_inference)

    def pause(self, step: int) -> str | None:
        if not self.enabled:
            return None

        request_id = None
        if self.world.is_master:
            get_logger().info(f"Waiting for inference pause before trainer checkpoint at step {step}")
            request_id = write_pause_request(self.output_dir, step)
            wait_for_marker(get_pause_step_dir(self.output_dir, step), PAUSED, request_id)
            get_logger().info(f"Inference paused for trainer checkpoint at step {step}")
        dist.barrier()
        return request_id

    def release(self, step: int, request_id: str | None) -> None:
        if not self.enabled:
            return
        if request_id is None:
            return
        if self.world.is_master:
            write_pause_release(self.output_dir, step, request_id)
