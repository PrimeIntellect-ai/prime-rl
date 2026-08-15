from pathlib import Path

from prime_rl.utils.config import BaseConfig


class WandbMonitorConfig(BaseConfig):
    project: str = "prime-rl"
    """W&B project to log to."""

    entity: str | None = None
    """W&B entity to log to."""

    name: str | None = None
    """W&B run name. Inherits ``run.name`` when unset."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool = False
    """Run W&B in offline mode."""


class FileMonitorConfig(BaseConfig):
    path: Path = Path("metrics.jsonl")
    """Path of the JSONL file, relative to the component's ``output_dir`` (absolute paths win)."""


class PrimeMonitorConfig(BaseConfig):
    name: str | None = None
    """Run name shown on the platform. Inherits ``run.name`` when unset."""


class MonitorsConfig(BaseConfig):
    wandb: WandbMonitorConfig | None = None
    """Log metrics to Weights & Biases. Off by default; enable with ``--monitors.wandb``."""

    file: FileMonitorConfig | None = FileMonitorConfig()
    """Log metrics and episode traces to the run's output directory. On by default; disable with ``--no-monitors.file``."""


class OrchestratorMonitorsConfig(MonitorsConfig):
    prime: PrimeMonitorConfig | None = None
    """Log metrics and episodes to the Prime Intellect platform. If None, disabled."""
