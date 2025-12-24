# prime_monitor/config.py
"""Configuration dataclasses for Prime Monitor."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MonitorConfig:
    """Configuration for the monitor application."""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    refresh_interval: float = 1.0
    trainer_gpu_ids: list[int] = field(default_factory=lambda: [0])
    inference_gpu_ids: list[int] = field(default_factory=lambda: [1])


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    index: int
    name: str
    utilization: float  # percentage
    memory_used: float  # MB
    memory_total: float  # MB
    temperature: float  # Celsius
    power_draw: float  # Watts


@dataclass
class TrainingMetrics:
    """Training metrics parsed from logs."""
    step: int = 0
    total_steps: int = 0
    throughput: float = 0.0  # tokens/s
    reward_mean: float = 0.0
    async_level: int = 0
    total_tokens: int = 0
    seq_len_mean: float = 0.0
    step_time: float = 0.0
    ckpt_step: int = 0
    last_update: str = ""