import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from prime_rl.configs.shared import SlurmConfig
from prime_rl.utils.config import find_package_resource
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from jinja2 import Template


def load_slurm_template(slurm: SlurmConfig) -> "Template":
    """Load ``slurm.template_path``. Includes resolve next to the template first, then in the
    bundled templates, so a custom template can include the bundled launch helpers."""
    from jinja2 import Environment, FileSystemLoader

    assert slurm.template_path is not None
    template_dirs = [slurm.template_path.parent]
    bundled_templates = find_package_resource("templates")
    if bundled_templates is not None:
        template_dirs.append(bundled_templates)
    env = Environment(loader=FileSystemLoader(template_dirs), keep_trailing_newline=True)
    return env.get_template(slurm.template_path.name)


def submit_sbatch(script_path: Path, log_message: str) -> None:
    """Submit ``script_path`` with sbatch and exit on failure."""
    logger = get_logger()
    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)
    logger.success(f"{result.stdout.strip()}\n\n{log_message}")
