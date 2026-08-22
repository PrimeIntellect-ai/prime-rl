from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.monitors import PrimeMonitorConfig, WandbMonitorConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.shared import MetricsServerConfig
from prime_rl.configs.trainer import TrainerConfig

ComponentRole = Literal["orchestrator", "trainer", "inference", "env-server"]
ComponentConfig: TypeAlias = OrchestratorConfig | TrainerConfig | InferenceConfig | EnvServerConfig

_CONFIG_TYPES: dict[ComponentRole, type[ComponentConfig]] = {
    "orchestrator": OrchestratorConfig,
    "trainer": TrainerConfig,
    "inference": InferenceConfig,
    "env-server": EnvServerConfig,
}


def _enabled(environ: MutableMapping[str, str], name: str) -> bool:
    value = environ.get(name)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{name} must be '1' when set, got {value!r}")
    return True


def _non_empty(environ: MutableMapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is not None and not value.strip():
        raise ValueError(f"{name} must not be empty when set")
    return value


def _override(
    owner: Any,
    attribute: str,
    value: Any,
    *,
    env_var: str,
    config_path: str,
    changes: list[str],
) -> None:
    previous = getattr(owner, attribute)
    if previous == value:
        return
    setattr(owner, attribute, value)
    changes.append(f"${env_var} overrides {config_path}: {previous!r} -> {value!r}")


def _metrics_port(environ: MutableMapping[str, str]) -> int | None:
    value = _non_empty(environ, "PRL_METRICS_PORT")
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"PRL_METRICS_PORT must be an integer between 1 and 65535, got {value!r}")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"PRL_METRICS_PORT must be between 1 and 65535, got {port}")
    return port


def _force_wandb(
    config: OrchestratorConfig | TrainerConfig,
    role: Literal["orchestrator", "trainer"],
    environ: MutableMapping[str, str],
    changes: list[str],
) -> None:
    if config.monitors.wandb is None:
        config.monitors.wandb = WandbMonitorConfig()
        changes.append("$PRL_FORCE_WANDB_MONITOR enables monitors.wandb")

    # An explicit offline monitor remains offline. Shared mode requires a server
    # connection and would otherwise silently override this config setting.
    if config.monitors.wandb.offline:
        return

    run_id = _non_empty(environ, "PRL_RUN_ID")
    if run_id is None:
        return

    shared_env = {
        "WANDB_SHARED_MODE": "1",
        "WANDB_RUN_ID": run_id,
        "WANDB_SHARED_LABEL": role,
    }
    if any(environ.get(name) != value for name, value in shared_env.items()):
        environ.update(shared_env)
        changes.append("$PRL_FORCE_WANDB_MONITOR configures this component for the shared W&B run")


def apply_platform_env(
    config: ComponentConfig,
    role: ComponentRole,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Apply the launcher-owned ``PRL_*`` contract to a resolved component config.

    The overlay runs after TOML/CLI parsing and before logging or monitor setup, so
    launcher wiring remains authoritative without coupling the config schema to the
    managed platform. Returned messages should be logged after the logger is set up.
    """
    expected_type = _CONFIG_TYPES.get(role)
    if expected_type is None:
        raise ValueError(f"Unknown component role {role!r}")
    if not isinstance(config, expected_type):
        raise TypeError(f"Role {role!r} requires {expected_type.__name__}, got {type(config).__name__}")

    environ = os.environ if environ is None else environ
    changes: list[str] = []

    # Validate the shared identity even in components that do not otherwise consume it.
    _non_empty(environ, "PRL_RUN_ID")

    if _enabled(environ, "PRL_JSON_LOGGING"):
        _override(
            config.log,
            "json_logging",
            True,
            env_var="PRL_JSON_LOGGING",
            config_path="log.json_logging",
            changes=changes,
        )

    if output_dir := _non_empty(environ, "PRL_OUTPUT_DIR"):
        _override(
            config,
            "output_dir",
            Path(output_dir),
            env_var="PRL_OUTPUT_DIR",
            config_path="output_dir",
            changes=changes,
        )

    if role == "orchestrator":
        assert isinstance(config, OrchestratorConfig)
        if _enabled(environ, "PRL_FORCE_PRIME_MONITOR") and config.monitors.prime is None:
            config.monitors.prime = PrimeMonitorConfig()
            changes.append("$PRL_FORCE_PRIME_MONITOR enables monitors.prime")
        if _enabled(environ, "PRL_FORCE_WANDB_MONITOR"):
            _force_wandb(config, role, environ, changes)
    elif role == "trainer":
        assert isinstance(config, TrainerConfig)
        if _enabled(environ, "PRL_FORCE_WANDB_MONITOR"):
            _force_wandb(config, role, environ, changes)
        if (port := _metrics_port(environ)) is not None:
            host = config.metrics_server.host if config.metrics_server is not None else "0.0.0.0"
            _override(
                config,
                "metrics_server",
                MetricsServerConfig(port=port, host=host),
                env_var="PRL_METRICS_PORT",
                config_path="metrics_server",
                changes=changes,
            )

    return changes
