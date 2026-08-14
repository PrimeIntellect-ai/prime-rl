"""Metric monitors.

Monitors are registered once per process via ``setup`` and used through the
module-level functions (``log``, ``log_episodes``, ...), which fan out to every
registered monitor. Fan-out never raises — a monitoring failure must not take
down training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.monitors import FileMonitorConfig, PrimeMonitorConfig, WandbConfig
from prime_rl.monitors.base import Monitor
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.prime import PrimeMonitor
from prime_rl.monitors.wandb import WandbMonitor
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

__all__ = [
    "Monitor",
    "WandbMonitor",
    "PrimeMonitor",
    "FileMonitor",
    "setup",
    "get",
    "log",
    "log_episodes",
    "finalize",
    "run_id",
]

_monitors: list[Monitor] = []


def setup(
    wandb: WandbConfig | None = None,
    prime: PrimeMonitorConfig | None = None,
    file: FileMonitorConfig | None = None,
    *,
    output_dir: Path,
    run_config: BaseConfig | None = None,
    train_env_names: list[str] | None = None,
    eval_env_names: list[str] | None = None,
) -> None:
    """Construct, initialize, and register one monitor per non-None config.

    Only rank 0 registers monitors — on other ranks the fan-out functions are
    no-ops. A monitor whose ``init`` raises crashes the run: a configured
    monitor must work. Prime registers first so ``run_id`` prefers the
    platform run id over W&B's.
    """
    assert not _monitors, "Monitors already set up. Call `setup` only once per process."
    rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
    if rank != 0:
        return

    monitors: list[tuple[Monitor, dict[str, Any]]] = []
    if prime is not None:
        monitors.append((PrimeMonitor(prime), dict(run_config=run_config)))
    if wandb is not None:
        monitors.append(
            (
                WandbMonitor(wandb),
                dict(
                    output_dir=output_dir,
                    run_config=run_config,
                    train_env_names=train_env_names,
                    eval_env_names=eval_env_names,
                ),
            )
        )
    if file is not None:
        monitors.append((FileMonitor(file), dict(output_dir=output_dir)))

    for monitor, init_kwargs in monitors:
        monitor.init(**init_kwargs)
        _monitors.append(monitor)


def get(monitor_cls: type[Monitor]) -> Monitor | None:
    """The registered monitor of the given type, None when it isn't running
    (not configured, or a non-zero rank)."""
    return next((monitor for monitor in _monitors if isinstance(monitor, monitor_cls)), None)


def run_id() -> str | None:
    """External run id of this run (platform run id when available, else W&B's)."""
    return next((monitor.run_id for monitor in _monitors if monitor.run_id), None)


def log(metrics: dict[str, Any], step: int) -> None:
    """Log scalar metrics for one step to all registered monitors."""
    for monitor in _monitors:
        try:
            monitor.log(metrics, step=step)
        except Exception as e:
            get_logger().warning(f"Failed to log metrics to {monitor.__class__.__name__}: {e}")


def log_episodes(rollouts: list[Rollout], step: int) -> None:
    """Log full episodes to all registered monitors that support it."""
    for monitor in _monitors:
        try:
            monitor.log_episodes(rollouts, step=step)
        except Exception as e:
            get_logger().warning(f"Failed to log episodes to {monitor.__class__.__name__}: {e}")


def finalize() -> None:
    """Finalize the run on all registered monitors."""
    for monitor in _monitors:
        try:
            monitor.finalize()
        except Exception as e:
            get_logger().warning(f"Failed to finalize {monitor.__class__.__name__}: {e}")
