import os
import subprocess
import sys
import uuid
from pathlib import Path

import tomli_w

from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import parse_argv


def write_trainer_config(config: SFTTrainerConfig, output_dir: Path) -> None:
    """Write resolved trainer config to disk, excluding launcher-only fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_data = config.model_dump(exclude={"deployment"}, exclude_none=True, mode="json")
    with open(output_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(trainer_data, f)


def render_slurm_script(config: SFTTrainerConfig, config_dir: Path) -> tuple[str, str]:
    """Render the SLURM script template. Returns (script, log_message)."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "single_node":
        config_path = config_dir / "sft.toml"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "wb") as f:
            tomli_w.dump(config.model_dump(exclude={"slurm", "deployment"}, exclude_none=True, mode="json"), f)

        script = template.render(
            config_path=config_path,
            output_dir=config.output_dir,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            gpus_per_node=config.deployment.gpus_per_node,
            partition=config.slurm.partition,
        )
        log_dir = config.output_dir / "logs"
        log_message = f"Logs:\n  Trainer:  tail -F {log_dir}/trainer/rank_0.log"
    else:
        script = template.render(
            config_dir=config_dir,
            output_dir=config.output_dir,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            num_nodes=config.deployment.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            partition=config.slurm.partition,
        )
        log_message = f"Logs:\n  Trainer:  tail -F {config.output_dir}/slurm/latest_train_node_rank_0.log"

    return script, log_message


def sft_slurm(config: SFTTrainerConfig):
    """Run SFT training via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"

    if config.deployment.type == "multi_node":
        write_trainer_config(config, config_dir)
        logger.info(f"Wrote trainer config to {config_dir}")

    script, log_message = render_slurm_script(config, config_dir)
    script_path = config.output_dir / "sft.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.slurm.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def sft_local(config: SFTTrainerConfig):
    """Run SFT training locally."""
    assert config.deployment.type == "single_node"

    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "trainer.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(config.model_dump(exclude={"deployment"}, exclude_none=True, mode="json"), f)

    trainer_cmd = [
        "uv",
        "run",
        "env",
        "PYTHONUNBUFFERED=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "torchrun",
        "--standalone",
        f"--nproc-per-node={config.deployment.num_gpus}",
        "-m",
        "prime_rl.trainer.sft.train",
        "@",
        config_path.as_posix(),
    ]

    result = subprocess.run(
        trainer_cmd,
        env={
            **os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    sys.exit(result.returncode)


def sft(config: SFTTrainerConfig):
    if config.slurm is not None:
        sft_slurm(config)
    else:
        sft_local(config)


def main():
    sft(parse_argv(SFTTrainerConfig))


if __name__ == "__main__":
    main()
