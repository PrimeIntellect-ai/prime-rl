import os
from pathlib import Path
from typing import Any

import torch
import verifiers as vf
from torch.utils.tensorboard import SummaryWriter

from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


class TensorBoardMonitor(Monitor):
    """Logs scalar metrics to TensorBoard."""

    def __init__(self, output_dir: Path):
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.is_master = rank == 0

        if not self.is_master:
            self.writer = None
            return

        tb_log_dir = output_dir / "tb_logs"
        self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        self.logger.info(f"TensorBoard logging enabled at: {tb_log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append(metrics)
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(key, value.item(), step)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        pass

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        pass

    def log_final_samples(self) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if self.writer is None:
            return
        for key, values in distributions.items():
            self.writer.add_histogram(key, torch.tensor(values), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
