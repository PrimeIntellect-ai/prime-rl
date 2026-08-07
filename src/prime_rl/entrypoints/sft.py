import os
import subprocess
import sys
from pathlib import Path

import tomli_w

from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.local_trainer import launch_local_trainer
from prime_rl.utils.config import cli, to_toml_dict
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import format_log_message, get_config_dir, get_log_dir, validate_output_dir
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
    set_proc_title,
)

SFT_TOML = "sft.toml"
SFT_SBATCH = "sft.sbatch"


def write_config(config: SFTConfig, config_path: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(to_toml_dict(config, exclude=exclude), f)


def write_slurm_script(config: SFTConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    trainer_env_vars = {
        **DEFAULT_COMMON_ENV_VARS,
        **DEFAULT_TRAINER_ENV_VARS,
        **config.env_vars,
    }

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            trainer_env_vars=trainer_env_vars,
            num_nodes=config.deployment.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            ranks_filter=",".join(map(str, config.log.ranks_filter)),
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def sft_slurm(config: SFTConfig):
    """Run SFT training via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / SFT_TOML
    exclude = (
        {"deployment", "slurm", "dry_run", "clean_output_dir"}
        if config.deployment.type == "multi_node"
        else {"slurm", "dry_run", "clean_output_dir"}
    )
    write_config(config, config_path, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / SFT_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_dir = get_log_dir(config.output_dir)
    num_nodes = config.deployment.num_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(log_dir=log_dir, trainer=True, num_train_nodes=num_nodes)

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def sft_local(config: SFTConfig):
    """Run SFT training locally with process monitoring and cleanup."""
    launch_local_trainer(
        config,
        config_filename=SFT_TOML,
        trainer_module="prime_rl.trainer.sft.train",
        display_name="SFT",
    )


def sft(config: SFTConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.model.name)

    if config.slurm is not None:
        sft_slurm(config)
    else:
        sft_local(config)


def main():
    set_proc_title("SFT")
    sft(cli(SFTConfig))


if __name__ == "__main__":
    main()
