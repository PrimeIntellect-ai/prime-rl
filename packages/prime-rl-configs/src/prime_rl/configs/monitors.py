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
    """Path of the metrics JSONL file, relative to the file monitor's directory under the
    component's ``output_dir`` (``monitors/file/``; absolute paths win)."""

    chunk_bytes: int = 5 * 1024**3
    """Size at which a trace stream (the episodes, each producer's annotations) rolls to
    a new numbered chunk file. A line never spans chunks."""

    compress: bool = True
    """Seal full chunks with zstd once the stream rolls past them (the live chunk stays
    plain text). Sealed chunks use seekable frames: a reader still lands on one line,
    and ``zstd -dcf stream/* | jq`` reads a whole stream."""

    float_decimals: int | None = 4
    """Decimals kept for per-token float streams (logprobs, entropies, advantages) in
    trace and annotation records; ``None`` keeps every digit. Training reads the wire,
    not the records, so this only trades record size against overlay precision."""


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
