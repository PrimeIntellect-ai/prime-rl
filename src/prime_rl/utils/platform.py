"""Prime Intellect platform integration for external training runs.

Registers a run with the platform before training starts and finalizes it on exit.
The run appears live in the platform dashboard and receives streamed metrics,
samples, and distributions via PrimeMonitor.
"""

import httpx
from prime_cli.core.config import Config as PrimeConfig

from prime_rl.configs.shared import PlatformConfig
from prime_rl.utils.logger import get_logger


def register_run(
    config: PlatformConfig,
    base_model: str,
    max_steps: int | None = None,
    environments: list[dict] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> tuple[str, str]:
    """Register an external training run with the platform.

    Args:
        config:        PlatformConfig with base_url and optional run metadata.
        base_model:    HuggingFace model name (e.g. "Qwen/Qwen3-4B").
        max_steps:     Total training steps (used for progress display).
        environments:  List of environment configs (e.g. [{"id": "reverse-text"}]).
        wandb_project: W&B project to display in the platform run metadata.
        wandb_entity:  W&B entity to display in the platform run metadata.

    Returns:
        Tuple of (run_id, monitoring_base_url). Pass run_id as RUN_ID env var and
        monitoring_base_url as PrimeMonitorConfig.base_url.

    Raises:
        RuntimeError: If the API call fails or no API key is available.
    """
    logger = get_logger()
    prime_config = PrimeConfig()

    api_key = prime_config.api_key or None
    if not api_key:
        raise RuntimeError(
            "Prime Intellect API key not found. Either:\n"
            "  • Set PRIME_API_KEY environment variable, or\n"
            "  • Run `prime login` to store credentials in ~/.prime/config.json"
        )

    team_id = config.team_id or prime_config.team_id

    payload: dict = {
        "base_model": base_model,
        "max_steps": max_steps or 0,
    }
    if config.run_name:
        payload["name"] = config.run_name
    if environments:
        payload["environments"] = environments
    if wandb_project:
        payload["wandb_project"] = wandb_project
    if wandb_entity:
        payload["wandb_entity"] = wandb_entity
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

    run_id = response.json()["run"]["id"]
    monitoring_base_url = f"{config.base_url}/api/internal/rft"
    dashboard_url = f"{config.base_url}/dashboard/training/{run_id}"
    logger.success(f"Monitor run at:\n  {dashboard_url}")
    return run_id, monitoring_base_url


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

    api_key = PrimeConfig().api_key or None
    if not api_key:
        logger.warning(f"Cannot finalize platform run {run_id}: no API key available.")
        return

    payload: dict = {"status": "completed" if success else "failed"}
    if error_message:
        payload["error_message"] = error_message

    status_label = "completed" if success else "failed"
    logger.info(f"Finalizing platform run {run_id} as {status_label}")

    try:
        response = httpx.put(
            f"{config.base_url}/api/v1/rft/external-runs/{run_id}/status",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        logger.warning(f"Failed to finalize platform run {run_id}: {exc}")
        return

    if response.status_code != 200:
        logger.warning(f"Failed to finalize platform run {run_id} (HTTP {response.status_code}): {response.text}")
        return

    logger.info(f"Platform run {run_id} marked as {status_label}")
