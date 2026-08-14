from pathlib import Path

from prime_rl.utils.config import BaseConfig


class WandbMonitorConfig(BaseConfig):
    project: str = "prime-rl"
    """W&B project to log to."""

    entity: str | None = None
    """W&B entity to log to."""

    name: str | None = None
    """W&B run name."""

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
    """Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one."""

    team_id: str | None = None
    """Team ID to associate the run with. Defaults to the Prime CLI team."""


class MonitorsConfig(BaseConfig):
    wandb: WandbMonitorConfig | None = None
    """Log metrics to Weights & Biases. If None, W&B logging is disabled."""

    file: FileMonitorConfig | None = None
    """Append metrics to a local JSONL file under the run's output directory. If None, disabled."""


class OrchestratorMonitorsConfig(MonitorsConfig):
    prime: PrimeMonitorConfig | None = None
    """Log metrics and episodes to the Prime Intellect platform. If None, disabled."""
