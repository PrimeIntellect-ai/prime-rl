"""Adapter discovery utilities for persisting LoRA adapters across inference server restarts.

This module provides functions to discover existing broadcast adapters from run folders
and format them for loading into vLLM at startup.
"""

from dataclasses import dataclass
from pathlib import Path

import tomli

from prime_rl.utils.logger import get_logger


@dataclass
class DiscoveredAdapter:
    """Represents a discovered LoRA adapter from a run folder."""

    name: str
    path: Path
    run_id: str
    step: int


def get_lora_name_from_config(config_path: Path) -> str | None:
    """Extract the LoRA adapter name from an orchestrator config file.

    Args:
        config_path: Path to orch.toml config file

    Returns:
        The LoRA name if found, None otherwise
    """
    logger = get_logger()

    if not config_path.exists():
        return None

    try:
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)

        # Navigate to model.lora.name in the config
        model_config = config_dict.get("model", {})
        lora_config = model_config.get("lora")

        if lora_config is None:
            return None

        return lora_config.get("name")
    except Exception as e:
        logger.warning(f"Failed to read LoRA name from {config_path}: {e}")
        return None


def get_latest_broadcast_step(broadcast_dir: Path) -> int | None:
    """Find the latest broadcast step that has a STABLE marker.

    Args:
        broadcast_dir: Path to the broadcasts directory

    Returns:
        The latest step number with a STABLE marker, or None if none found
    """
    if not broadcast_dir.exists():
        return None

    step_dirs = list(broadcast_dir.glob("step_*"))
    if not step_dirs:
        return None

    # Filter to only steps with STABLE marker and extract step numbers
    stable_steps = []
    for step_dir in step_dirs:
        if (step_dir / "STABLE").exists():
            try:
                step_num = int(step_dir.name.split("_")[-1])
                stable_steps.append(step_num)
            except ValueError:
                continue

    if not stable_steps:
        return None

    return max(stable_steps)


def validate_adapter_path(adapter_path: Path) -> bool:
    """Validate that an adapter path contains valid adapter files.

    Args:
        adapter_path: Path to the adapter directory

    Returns:
        True if the adapter path contains valid adapter files
    """
    # Check for adapter_config.json (required for PEFT-compatible adapters)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return False

    # Check for adapter weights (either safetensors or torch format)
    has_weights = (
        list(adapter_path.glob("*.safetensors"))
        or list(adapter_path.glob("*.bin"))
        or list(adapter_path.glob("*.pt"))
    )

    return bool(has_weights)


def discover_adapters(output_dir: Path) -> list[DiscoveredAdapter]:
    """Discover all valid LoRA adapters from run folders in the output directory.

    This function scans the output directory for run folders, reads their orchestrator
    configs to find the LoRA adapter name, and locates the latest broadcast adapter
    that is ready (has STABLE marker).

    Args:
        output_dir: Path to the output directory containing run folders

    Returns:
        List of DiscoveredAdapter objects for each valid adapter found
    """
    logger = get_logger()
    discovered = []

    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return discovered

    # Find all run directories
    run_dirs = list(output_dir.glob("run_*"))
    logger.debug(f"Found {len(run_dirs)} run directories in {output_dir}")

    for run_dir in run_dirs:
        run_id = run_dir.name

        # Get the LoRA name from the config
        config_path = run_dir / "configs" / "orch.toml"
        lora_name = get_lora_name_from_config(config_path)

        if lora_name is None:
            logger.debug(f"No LoRA config found for run {run_id}, skipping")
            continue

        # Find the latest broadcast step
        broadcast_dir = run_dir / "broadcasts"
        latest_step = get_latest_broadcast_step(broadcast_dir)

        if latest_step is None:
            logger.debug(f"No stable broadcast found for run {run_id}, skipping")
            continue

        # Construct the adapter path
        adapter_path = broadcast_dir / f"step_{latest_step}"

        # Validate the adapter
        if not validate_adapter_path(adapter_path):
            logger.warning(
                f"Adapter at {adapter_path} appears invalid (missing adapter_config.json or weights), skipping"
            )
            continue

        discovered.append(
            DiscoveredAdapter(
                name=lora_name,
                path=adapter_path,
                run_id=run_id,
                step=latest_step,
            )
        )
        logger.info(f"Discovered adapter '{lora_name}' for run {run_id} at step {latest_step}")

    return discovered


def format_lora_modules_arg(adapters: list[DiscoveredAdapter]) -> list[str]:
    """Format discovered adapters as vLLM --lora-modules argument values.

    vLLM expects --lora-modules in the format: name=path name2=path2
    This function returns a list of "name=path" strings that can be passed to vLLM.

    Args:
        adapters: List of discovered adapters

    Returns:
        List of "name=path" strings for the --lora-modules argument
    """
    lora_modules = []
    seen_names = set()

    for adapter in adapters:
        # Skip duplicate adapter names (only load the first one found)
        if adapter.name in seen_names:
            logger = get_logger()
            logger.warning(
                f"Duplicate adapter name '{adapter.name}' found, keeping first occurrence"
            )
            continue

        seen_names.add(adapter.name)
        # Format as "name=path" for vLLM
        lora_modules.append(f"{adapter.name}={adapter.path}")

    return lora_modules
