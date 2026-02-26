"""Prime Intellect platform integration for external training runs.

Registers a run with the platform before training starts and finalizes it on exit.
The run appears live in the platform dashboard and receives streamed metrics,
samples, and distributions via PrimeMonitor.
"""

import json
import os
from pathlib import Path

import httpx

from prime_rl.configs.shared import PlatformConfig
from prime_rl.utils.logger import get_logger


def _read_prime_config() -> dict:
    """Read ~/.prime/config.json written by `prime login`. Returns {} if absent."""
    config_path = Path.home() / ".prime" / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _resolve_api_key() -> str | None:
    """Resolve the Prime Intellect API key from PRIME_API_KEY env var or ~/.prime/config.json."""
    return os.getenv("PRIME_API_KEY") or _read_prime_config().get("api_key")


def register_run(
    config: PlatformConfig,
    base_model: str,
    max_steps: int | None = None,
    environments: list[dict] | None = None,
) -> str:
    """Register an external training run with the platform and return the run ID.

    Args:
        config:       PlatformConfig with base_url and optional run metadata.
        base_model:   HuggingFace model name (e.g. "Qwen/Qwen3-4B").
        max_steps:    Total training steps (used for progress display).
        environments: List of environment configs (e.g. [{"id": "reverse-text"}]).

    Returns:
        The run ID string. Set as RUN_ID env var for PrimeMonitor.

    Raises:
        RuntimeError: If the API call fails or no API key is available.
    """
    logger = get_logger()

    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "Prime Intellect API key not found. Either:\n"
            "  • Set PRIME_API_KEY environment variable, or\n"
            "  • Run `prime login` to store credentials in ~/.prime/config.json"
        )

    team_id = config.team_id or _read_prime_config().get("team_id")

    payload: dict = {
        "base_model": base_model,
        "max_steps": max_steps or 0,
    }
    if config.run_name:
        payload["name"] = config.run_name
    if environments:
        payload["environments"] = environments
    if config.wandb_project:
        payload["wandb_project"] = config.wandb_project
    if config.wandb_entity:
        payload["wandb_entity"] = config.wandb_entity
    if team_id:
        payload["team_id"] = team_id

    logger.info(f"Registering external training run with platform at {config.base_url}")

    response = httpx.post(
        f"{config.base_url}/api/v1/rft/external-runs",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )

    if response.status_code != 201:
        raise RuntimeError(f"Failed to create platform run (HTTP {response.status_code}): {response.text}")

    run_id = response.json()["run_id"]
    logger.info(f"Platform run registered: {config.base_url}/dashboard/training/{run_id}")
    return run_id


def finalize_run(
    config: PlatformConfig,
    run_id: str,
    success: bool,
    error_message: str | None = None,
) -> None:
    """Mark an external run as completed or failed on the platform.

    Args:
        config:        PlatformConfig with base_url.
        run_id:        The run ID returned by register_run.
        success:       True if training completed successfully, False if it failed.
        error_message: Optional error message when success=False.
    """
    logger = get_logger()

    api_key = _resolve_api_key()
    if not api_key:
        logger.warning(f"Cannot finalize platform run {run_id}: no API key available.")
        return

    payload: dict = {"status": "completed" if success else "failed"}
    if error_message:
        payload["error_message"] = error_message

    status_label = "completed" if success else "failed"
    logger.info(f"Finalizing platform run {run_id} as {status_label}")

    response = httpx.put(
        f"{config.base_url}/api/v1/rft/external-runs/{run_id}/status",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        logger.warning(f"Failed to finalize platform run {run_id} (HTTP {response.status_code}): {response.text}")
        return

    logger.info(f"Platform run {run_id} marked as {status_label}")
