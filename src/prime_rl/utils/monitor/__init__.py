import os
from pathlib import Path

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import PrimeMonitorConfig, WandbWithExtrasConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.monitor.base import Monitor, NoOpMonitor
from prime_rl.utils.monitor.file import FileMonitor
from prime_rl.utils.monitor.multi import MultiMonitor
from prime_rl.utils.monitor.prime import PrimeMonitor
from prime_rl.utils.monitor.wandb import WandbMonitor

SWEEP_METRICS_JSONL_ENV = "PRIME_RL_SWEEP_METRICS_JSONL"

__all__ = [
    "FileMonitor",
    "Monitor",
    "WandbMonitor",
    "PrimeMonitor",
    "MultiMonitor",
    "NoOpMonitor",
    "SWEEP_METRICS_JSONL_ENV",
    "setup_monitor",
    "get_monitor",
]

_MONITOR: Monitor | None = None


def get_monitor() -> Monitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("Monitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    wandb_config: WandbWithExtrasConfig | None = None,
    output_dir: Path | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseConfig | None = None,
    *,
    prime_config: PrimeMonitorConfig | None = None,
    keep_full_history: bool = True,
    # Backward compatibility: support old 'config' keyword argument
    config: WandbWithExtrasConfig | None = None,
) -> Monitor:
    """
    Sets up monitors to log metrics.

    `keep_full_history`: when False, monitors retain only the most recent
    metrics dict. The orchestrator passes False outside `--bench` mode to
    avoid an unbounded list growing for the lifetime of the run.
    """
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("Monitor already initialized. Please call `setup_monitor` only once.")

    if config is not None and wandb_config is None:
        wandb_config = config

    monitors: list[Monitor] = []

    if wandb_config is not None:
        monitors.append(
            WandbMonitor(
                config=wandb_config,
                output_dir=output_dir,
                tokenizer=tokenizer,
                run_config=run_config,
                keep_full_history=keep_full_history,
            )
        )

    if prime_config is not None:
        monitors.append(
            PrimeMonitor(
                config=prime_config,
                output_dir=output_dir,
                tokenizer=tokenizer,
                run_config=run_config,
                keep_full_history=keep_full_history,
            )
        )

    sweep_metrics_path = os.environ.get(SWEEP_METRICS_JSONL_ENV)
    if sweep_metrics_path:
        # Sweep launcher injects the path; trial subprocess inherits it. Off
        # in non-sweep runs because the env var is unset.
        monitors.append(
            FileMonitor(
                output_path=Path(sweep_metrics_path),
                keep_full_history=keep_full_history,
            )
        )

    if len(monitors) == 0:
        _MONITOR = NoOpMonitor(keep_full_history=keep_full_history)
    elif len(monitors) == 1:
        _MONITOR = monitors[0]
    else:
        _MONITOR = MultiMonitor(monitors)

    return _MONITOR
