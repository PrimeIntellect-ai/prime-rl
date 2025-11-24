from pathlib import Path

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import PrimeMonitorWithExtrasConfig, WandbWithExtrasConfig
from prime_rl.utils.monitor.base import Monitor, NoOpMonitor
from prime_rl.utils.monitor.multi import MultiMonitor
from prime_rl.utils.monitor.prime import PrimeMonitor
from prime_rl.utils.monitor.wandb import WandbMonitor
from prime_rl.utils.pydantic_config import BaseSettings

__all__ = [
    "Monitor",
    "WandbMonitor",
    "PrimeMonitor",
    "MultiMonitor",
    "NoOpMonitor",
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
    run_config: BaseSettings | None = None,
    *,
    prime_config: PrimeMonitorWithExtrasConfig | None = None,
    # Backward compatibility: support old 'config' keyword argument
    config: WandbWithExtrasConfig | None = None,
) -> Monitor:
    """Sets up monitors to log metrics.

    Args:
        wandb_config: Optional W&B monitor configuration (can be passed positionally for backward compatibility)
        output_dir: Optional output directory for monitors
        tokenizer: Optional tokenizer for sample logging
        run_config: Optional run configuration to log
        prime_config: Optional Prime Intellect monitor configuration (keyword-only)
        config: Optional W&B monitor configuration (old style keyword argument, for backward compatibility)

    Returns:
        Monitor instance (MultiMonitor if multiple configs provided, single monitor if one, NoOpMonitor if none)
    """
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("Monitor already initialized. Please call `setup_monitor` only once.")

    # Backward compatibility: if config is provided (old style), use it as wandb_config
    if config is not None and wandb_config is None:
        wandb_config = config

    monitors: list[Monitor] = []

    # Create W&B monitor if configured
    if wandb_config is not None:
        monitors.append(
            WandbMonitor(
                config=wandb_config,
                output_dir=output_dir,
                tokenizer=tokenizer,
                run_config=run_config,
            )
        )

    # Create Prime monitor if configured
    if prime_config is not None:
        monitors.append(
            PrimeMonitor(
                config=prime_config,
                output_dir=output_dir,
                tokenizer=tokenizer,
                run_config=run_config,
            )
        )

    # Return appropriate monitor
    if len(monitors) == 0:
        _MONITOR = NoOpMonitor()
    elif len(monitors) == 1:
        _MONITOR = monitors[0]
    else:
        _MONITOR = MultiMonitor(monitors)

    return _MONITOR

