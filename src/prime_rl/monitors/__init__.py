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
from typing import TYPE_CHECKING, Any, Literal, overload

from prime_rl.configs.monitors import FileMonitorConfig, PrimeMonitorConfig, WandbMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, MonitorContext
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.prime import PrimeMonitor
from prime_rl.monitors.wandb import WandbMonitor
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import format_time, get_logger

if TYPE_CHECKING:
    import verifiers.v1 as vf

__all__ = [
    "Monitor",
    "MonitorContext",
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

    context = MonitorContext(
        output_dir=output_dir,
        producer=producer,
        run_config=run_config,
        train_env_names=train_env_names or [],
        eval_env_names=eval_env_names or [],
        overview_flavor=overview_flavor,
    )
    configured = [(PrimeMonitor, prime), (WandbMonitor, wandb), (FileMonitor, file)]
    monitors = [cls(config, context) for cls, config in configured if config is not None]
    if not monitors:
        return

    get_logger().info(f"Initializing monitors ({', '.join(type(monitor).__name__ for monitor in monitors)})")
    t0 = time.perf_counter()
    for monitor in monitors:
        get_logger().info(f"Initializing {type(monitor).__name__} ({monitor.config})")
        await monitor.init()
        MONITORS.append(monitor)
    get_logger().debug(f"Initialized monitors in {format_time(time.perf_counter() - t0)}")


def get(monitor_cls: type[Monitor]) -> Monitor | None:
    """The registered monitor of the given type, None when it isn't running
    (not configured, or a non-zero rank)."""
    return next((monitor for monitor in MONITORS if isinstance(monitor, monitor_cls)), None)


async def _fan_out(hook: str, *args: Any, **kwargs: Any) -> None:
    for monitor in MONITORS:
        try:
            await getattr(monitor, hook)(*args, **kwargs)
        except Exception as e:
            get_logger().warning(f"Failed to {hook} on {type(monitor).__name__}: {e}")


@overload
async def log(data: dict[str, Any], step: int | None) -> None: ...


@overload
async def log(data: vf.Episode | list[vf.Episode], step: int, kind: Kind) -> None: ...


async def log(data: dict[str, Any] | vf.Episode | list[vf.Episode], step: int | None, kind: Kind = "train") -> None:
    """Log to all registered monitors: a dict of scalar metrics, or episodes as they
    arrive."""
    await _fan_out("log", data, step=step, kind=kind)


async def log_annotations(updates: list[dict[str, Any]]) -> None:
    """Log trace updates to all registered monitors."""
    await _fan_out("log_annotations", updates)


async def log_eval_plan(env_name: str, step: int, expected: int) -> None:
    """Log an eval epoch's expected episode count to all registered monitors."""
    await _fan_out("log_eval_plan", env_name, step, expected)


async def log_live(events: list[dict[str, Any]]) -> None:
    """Log live-trace events to all registered monitors."""
    await _fan_out("log_live", events)


async def log_eval_epoch(env_name: str, step: int, episodes: list[vf.Episode]) -> None:
    """Log one finished eval epoch to all registered monitors."""
    await _fan_out("log_eval_epoch", env_name, step, episodes)


async def finalize(paused: bool = False) -> None:
    """Finalize the run on all registered monitors. A paused run is not reported as completed."""
    await _fan_out("finalize", paused=paused)
