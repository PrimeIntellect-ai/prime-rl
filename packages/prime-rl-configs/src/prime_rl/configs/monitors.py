from pathlib import Path
from typing import Annotated

from pydantic import Field

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
    and ``zstd -dcf stream/*.jsonl* | jq`` reads a whole stream."""

    float_decimals: int | None = 4
    """Decimals kept for per-token float streams (logprobs, entropies, advantages) in
    trace and annotation records; ``None`` keeps every digit. Training reads the wire,
    not the records, so this only trades record size against overlay precision."""

    live: bool = True
    """Write in-flight traces under ``traces/live/`` for the dashboard's live view: one file
    per trace, appended as each turn streams, plus a placeholder per dispatched episode. Off,
    an episode is written once, when it finishes (the dashboard shows finished episodes only),
    which spares the run directory a small append per turn of every in-flight rollout."""

    trace_sample: Annotated[float, Field(gt=0, le=1)] = 1.0
    """Fraction of train rollout groups whose traces are written: their episodes in the
    trace stream, their live traces, and the annotations of the process that logs the
    episodes (the orchestrator). A group is in or out as a whole, decided by a hash of its
    id, so every written group is complete. Eval episodes and metrics are always written.
    Annotations from a process that logs no episodes (the trainer's ``annotations``) are
    not sampled. Lower it to cut the run directory's write load on a busy run."""


class PrimeMonitorConfig(BaseConfig):
    name: str | None = None
    """Run or evaluation name shown on Prime. Inherits ``run.name`` when unset."""


class MonitorsConfig(BaseConfig):
    wandb: WandbMonitorConfig | None = None
    """Log metrics to Weights & Biases. Off by default; enable with ``--monitors.wandb``."""

    file: FileMonitorConfig | None = FileMonitorConfig()
    """Log metrics and episode traces to the run's output directory. On by default; disable with ``--no-monitors.file``."""

    prime: PrimeMonitorConfig | None = PrimeMonitorConfig()
    """Log metrics and episodes to the Prime Intellect platform: a training run's processes
    log to one platform run, and a standalone eval uploads each source's epoch as an
    evaluation. On by default and needs a Prime API key; disable with ``--no-monitors.prime``."""
