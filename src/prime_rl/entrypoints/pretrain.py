import json
import os
import uuid
from pathlib import Path

from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.entrypoints.sft import submit_slurm_script, write_trainer_slurm_script
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import (
    format_config_message,
    format_log_message,
    get_launcher_dir,
    prepare_attempt_dirs,
    validate_run_dir,
    write_launch_artifacts,
)
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
    ProcessGroup,
    set_proc_title,
    torchrun_cmd,
)

PRETRAIN_CONFIG = "pretrain.json"
PRETRAIN_SBATCH = "pretrain.sbatch"


def pretrain_slurm(config: PretrainConfig, config_path: Path, log_dir: Path) -> None:
    logger = get_logger()
    script_path = get_launcher_dir(config.run_dir) / PRETRAIN_SBATCH
    write_trainer_slurm_script(config, "pretrain", config_path, log_dir, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")
    num_nodes = config.deployment.num_train_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(log_dir=log_dir, trainer=True, num_train_nodes=num_nodes)
    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return
    submit_slurm_script(script_path, log_message, logger)


def pretrain(config: PretrainConfig):
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    os.environ.setdefault("PRL_RUN_ID", uuid.uuid4().hex)
    assert config.run.name is not None
    os.environ["PRL_RUN_NAME"] = config.run.name

    resuming = config.resume is not None
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_run_dir(
        config.run_dir,
        output_dir=config.output_dir,
        resuming=resuming,
        clean=config.clean,
        ckpt_output_dir=ckpt_output_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pretraining run")
    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    write_launch_artifacts(config_dir, "pretrain")

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.model.name, skip_weights=True)

    config_path = config_dir / PRETRAIN_CONFIG
    exclude = {"dry_run", "clean"}
    if config.slurm is not None:
        # A single-node job runs this launcher again on the node; the trainers of a
        # multi-node job read the config directly.
        exclude |= {"slurm", "deployment"} if config.deployment.type == "multi_node" else {"slurm"}
    config_path.write_text(json.dumps(dump_resolved_config(config, exclude=exclude), indent=2))
    logger.info(f"Configs:\n{format_config_message(config_dir, 'pretrain', [('Pretrain', config_path)])}")
    if config.slurm is not None:
        pretrain_slurm(config, config_path, log_dir)
        return
    if config.dry_run:
        logger.success("Dry run complete. To start the run, remove --dry-run from your command.")
        return

    logger.info(format_log_message(log_dir=log_dir, trainer=True))
    with ProcessGroup() as processes:
        processes.start(
            "trainer",
            torchrun_cmd(
                "prime_rl.trainer.pretrain.train",
                config_path,
                config.deployment.num_train_gpus,
                log_dir,
                config.log.ranks_filter,
            ),
            env={**os.environ, **DEFAULT_COMMON_ENV_VARS, **DEFAULT_TRAINER_ENV_VARS, **config.env_vars},
            log_path=log_dir / "trainer.log",
        )
        logger.success(f"Started trainer with {config.deployment.num_train_gpus} GPU(s)")
        processes.wait("trainer")
        logger.success("Pretraining finished!")


def main():
    set_proc_title("Pretrain")
    pretrain(cli(PretrainConfig))


if __name__ == "__main__":
    main()
