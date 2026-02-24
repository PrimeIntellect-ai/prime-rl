import subprocess
import sys
from pathlib import Path

import tomli_w
from jinja2 import Environment, FileSystemLoader
from pydantic import Field

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import BaseSettings, parse_argv

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "infer_slurm.sh.j2"


class InferSLURMConfig(BaseSettings):
    inference: InferenceConfig = InferenceConfig()

    job_name: str = "infer"
    output_dir: Path = Path("outputs/inference")
    gpus_per_node: int = Field(default=8, description="Number of GPUs per node.")

    project_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Path to the project root.",
    )

    slurm_template: Path | None = Field(
        default=None, description="Path to a custom SLURM template. If None, uses the default."
    )
    dry_run: bool = Field(default=False, description="Only generate the script without submitting.")


def write_inference_config(config: InferSLURMConfig, config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "inference.toml", "wb") as f:
        tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)


def render_slurm_script(config: InferSLURMConfig, config_dir: Path) -> str:
    if config.slurm_template is not None:
        template_dir = config.slurm_template.parent
        template_name = config.slurm_template.name
    else:
        template_dir = TEMPLATE_DIR
        template_name = TEMPLATE_NAME
    env = Environment(loader=FileSystemLoader(template_dir), keep_trailing_newline=True)
    template = env.get_template(template_name)
    return template.render(
        job_name=config.job_name,
        output_dir=config.output_dir,
        config_dir=config_dir,
        project_dir=config.project_dir,
        gpus_per_node=config.gpus_per_node,
    )


def infer_slurm(config: InferSLURMConfig):
    logger = setup_logger("info")

    config_dir = config.output_dir / "configs"
    write_inference_config(config, config_dir)
    logger.info(f"Wrote inference config to {config_dir}")

    script = render_slurm_script(config, config_dir)
    script_path = config.output_dir / "infer.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_message = f"Logs:\n  Inference:  tail -f {config.output_dir}/slurm/latest_infer.log"

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def main():
    infer_slurm(parse_argv(InferSLURMConfig))


if __name__ == "__main__":
    main()
