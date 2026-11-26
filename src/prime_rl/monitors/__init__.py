"""Metric monitors.

Monitors are registered once per process via ``setup`` and used through the
module-level functions (``log``, ``finalize``, ...), which fan out to every
registered monitor. Fan-out never raises — a monitoring failure must not take
down training.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prime_rl.configs.monitors import (
    FileMonitorConfig,
    PrimeMonitorConfig,
    WandbMonitorConfig,
)
from prime_rl.monitors.base import Kind, Monitor
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.prime import EVAL_ID_VAR, PrimeMonitor
from prime_rl.monitors.wandb import WandbMonitor
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import format_time, get_logger

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, EpisodeRecord
    from prime_rl.env.types import RecordRequest

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
    producer: str | None = None,
    run_config: BaseConfig | None = None,
    train_env_names: list[str] | None = None,
    eval_env_names: list[str] | None = None,
    overview_flavor: Literal["rl", "sft", "eval"] = "rl",
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

    # A platform launch creates the run ($RUN_ID, or a hosted eval's $PRIME_RUNS_EVAL_ID) and builds
    # the component configs with the default, which leaves the monitor off: log to that run anyway.
    if prime is None and (os.getenv("RUN_ID") or os.getenv(EVAL_ID_VAR)):
        prime = PrimeMonitorConfig()

    monitors: list[tuple[str, Monitor, dict[str, Any]]] = []
    if prime is not None:
        monitors.append(
            ("prime", PrimeMonitor(prime), dict(config=run_config, output_dir=output_dir, producer=producer))
        )
    if wandb is not None:
        monitors.append(
            (
                "wandb",
                WandbMonitor(wandb),
                dict(
                    output_dir=output_dir,
                    config=run_config,
                    train_env_names=train_env_names,
                    eval_env_names=eval_env_names,
                    overview_flavor=overview_flavor,
                ),
            )
        )
    if file is not None:
        monitors.append(("file", FileMonitor(file), dict(output_dir=output_dir, producer=producer)))
    if not monitors:
        return

    get_logger().info(f"Initializing monitors ({', '.join(name for name, _, _ in monitors)})")
    t0 = time.perf_counter()
    for name, monitor, init_kwargs in monitors:
        get_logger().info(f"Initializing {name} monitor ({monitor.config})")
        await monitor.init(**init_kwargs)
        MONITORS.append(monitor)
    get_logger().debug(f"Initialized monitors in {format_time(time.perf_counter() - t0)}")


def get(monitor_cls: type[Monitor]) -> Monitor | None:
    """The registered monitor of the given type, None when it isn't running
    (not configured, or a non-zero rank)."""
    return next((monitor for monitor in MONITORS if isinstance(monitor, monitor_cls)), None)


async def log(metrics: dict[str, Any], step: int | None) -> None:
    """Log a dict of scalar metrics to all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log(metrics, step=step)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


def record_request(kind: Kind, group_id: str | None) -> RecordRequest | None:
    """What a group's episodes must carry back for the monitors that persist them: their
    records, serialized as the file monitor writes them; None if no monitor persists them."""
    from prime_rl.env.types import RecordRequest

    wanting = [monitor for monitor in MONITORS if monitor.wants_record(kind, group_id)]
    if not wanting:
        return None
    file = next((monitor for monitor in wanting if isinstance(monitor, FileMonitor)), None)
    return RecordRequest(float_decimals=file.config.float_decimals) if file is not None else RecordRequest()


def wants_live(kind: Kind, group_id: str | None) -> bool:
    """Whether a monitor shows a group's in-flight traces whole."""
    return any(monitor.wants_live(kind, group_id) for monitor in MONITORS)


async def log_records(records: list[tuple[EpisodeHead, EpisodeRecord]], step: int, kind: Kind) -> None:
    """Persist finished episodes from their records on all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log_records(records, step=step, kind=kind)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def log_annotations(updates: list[dict[str, Any]]) -> None:
    """Log trace updates to all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log_annotations(updates)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def log_eval_plan(env_name: str, step: int, expected: int) -> None:
    """Log an eval epoch's expected episode count to all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log_eval_plan(env_name, step, expected)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def log_live(events: list[dict[str, Any]]) -> None:
    """Log live-trace events to all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log_live(events)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def log_eval_epoch(env_name: str, step: int, episodes: list[EpisodeHead]) -> None:
    """Log one finished eval epoch to all registered monitors."""
    for monitor in MONITORS:
        try:
            await monitor.log_eval_epoch(env_name, step, episodes)
        except Exception as e:
            get_logger().warning(f"Failed to log to {monitor.__class__.__name__}: {e}")


async def finalize(paused: bool = False) -> None:
    """Finalize the run on all registered monitors. A paused run is not reported as completed."""
    for monitor in MONITORS:
        try:
            await monitor.finalize(paused=paused)
        except Exception as e:
            get_logger().warning(f"Failed to finalize {monitor.__class__.__name__}: {e}")
