"""Metric monitors.

Monitors are registered once per process via ``setup`` and used through the
module-level functions (``log``, ``finalize``, ...), which fan out to every
registered monitor. Fan-out never raises — a monitoring failure must not take
down training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from prime_rl.configs.monitors import FileMonitorConfig, PrimeMonitorConfig, WandbMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.prime import PrimeMonitor
from prime_rl.monitors.wandb import WandbMonitor
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import verifiers.v1 as vf

__all__ = [
    "Monitor",
    "WandbMonitor",
    "PrimeMonitor",
    "FileMonitor",
    "setup",
    "get",
    "log",
    "finalize",
]

# All monitors registered for the current run.
MONITORS: list[Monitor] = []


async def setup(
    wandb: WandbMonitorConfig | None = None,
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
    monitor must work.
    """
    assert not MONITORS, "Monitors already set up. Call `setup` only once per process."
    rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
    if rank != 0:
        return

    monitors: list[tuple[Monitor, dict[str, Any]]] = []
    if prime is not None:
        monitors.append((PrimeMonitor(prime), dict(config=run_config)))
    if wandb is not None:
        monitors.append(
            (
                WandbMonitor(wandb),
                dict(
                    output_dir=output_dir,
                    config=run_config,
                    train_env_names=train_env_names,
                    eval_env_names=eval_env_names,
                ),
            )
        )
    if file is not None:
        monitors.append((FileMonitor(file), dict(output_dir=output_dir)))

    for monitor, init_kwargs in monitors:
        await monitor.init(**init_kwargs)
        MONITORS.append(monitor)


def get(monitor_cls: type[Monitor]) -> Monitor | None:
    """The registered monitor of the given type, None when it isn't running
    (not configured, or a non-zero rank)."""
    return next((monitor for monitor in MONITORS if isinstance(monitor, monitor_cls)), None)


@overload
async def log(data: dict[str, Any], step: int) -> None: ...


@overload
async def log(data: vf.Episode | list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None: ...


async def log(
    data: dict[str, Any] | vf.Episode | list[vf.Episode],
    step: int,
    kind: Kind = "train",
    subset: Subset = "effective",
) -> None:
    """Log to all registered monitors: a dict of scalar metrics, or episodes
    with their cohort coordinates (train/eval x all/effective)."""
    for monitor in MONITORS:
        try:
            await monitor.log(data, step=step, kind=kind, subset=subset)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def finalize() -> None:
    """Finalize the run on all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.finalize()
        except Exception as e:
            get_logger().warning(f"Failed to finalize {monitor.__class__.__name__}: {e}")
