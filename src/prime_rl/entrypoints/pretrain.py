import json
import os
import uuid
from pathlib import Path

from prime_rl.configs.pretrain import EvalWorkerConfig, PretrainConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.entrypoints.sft import submit_slurm_script, write_trainer_slurm_script
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import (
    format_config_message,
    format_log_message,
    get_launcher_dir,
    get_wandb_run_id_path,
    prepare_attempt_dirs,
    validate_run_dir,
    write_launch_artifacts,
)
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
    ProcessGroup,
    partition_gpus,
    set_proc_title,
    torchrun_cmd,
)

PRETRAIN_CONFIG = "pretrain.json"
PRETRAIN_SBATCH = "pretrain.sbatch"
EVAL_WORKER_CONFIG = "eval_worker.json"


def write_model_dir(config: PretrainConfig) -> None:
    """Write the config built from ``[model.arch]`` and the tokenizer to ``<run_dir>/model``. The vocabulary size
    (rounded up to a multiple of 128) and the special token ids come from the tokenizer."""
    from transformers import AutoTokenizer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    import prime_rl.trainer.models  # noqa: F401 - registers the custom model configs with AutoConfig

    arch = dict(config.model.arch)
    config_class = CONFIG_MAPPING[arch.pop("model_type")]
    # PretrainedConfig keeps unknown kwargs as attributes, so a typo would silently do nothing.
    if unknown := arch.keys() - config_class().to_dict().keys():
        raise ValueError(f"[model.arch] fields {sorted(unknown)} do not exist on {config_class.__name__}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.name, trust_remote_code=config.tokenizer.trust_remote_code
    )
    hf_config = config_class(
        **arch,
        vocab_size=-(-len(tokenizer) // 128) * 128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    model_dir = Path(config.model.name)
    hf_config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def write_eval_worker_config(config: PretrainConfig, config_dir: Path) -> Path:
    """The eval worker's config: the ``[eval_worker]`` suite for this run's directory."""
    assert config.eval_worker is not None
    suite = config.eval_worker.model_dump(exclude={"num_gpus", "num_nodes"})
    worker = EvalWorkerConfig(**suite, run_dir=config.run_dir, log=LogConfig(level=config.log.level))
    path = config_dir / EVAL_WORKER_CONFIG
    path.write_text(json.dumps(dump_resolved_config(worker), indent=2))
    return path


def pretrain_slurm(config: PretrainConfig, config_path: Path, log_dir: Path) -> None:
    logger = get_logger()
    script_path = get_launcher_dir(config.run_dir) / PRETRAIN_SBATCH
    eval_worker_vars = {}
    if config.eval_worker is not None and config.deployment.type == "multi_node":
        eval_worker_vars = dict(
            num_eval_nodes=config.eval_worker.num_nodes, eval_worker_config_path=config_path.parent / EVAL_WORKER_CONFIG
        )
    write_trainer_slurm_script(config, "pretrain", config_path, log_dir, script_path, **eval_worker_vars)
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
        write_model_dir(config)
    if config.eval_snapshots is not None and not get_wandb_run_id_path(config.run_dir).exists():
        # The trainer and the eval worker (a separate job) log into one W&B run; a resumed run keeps its id.
        get_wandb_run_id_path(config.run_dir).write_text(os.environ["PRL_RUN_ID"])

    config_path = config_dir / PRETRAIN_CONFIG
    exclude = {"dry_run", "clean"}
    if config.slurm is not None:
        # A single-node job runs this launcher again on the node; the trainers of a
        # multi-node job read the config directly.
        exclude |= {"slurm", "deployment"} if config.deployment.type == "multi_node" else {"slurm"}
    config_path.write_text(json.dumps(dump_resolved_config(config, exclude=exclude), indent=2))
    components = [("Pretrain", config_path)]
    if config.eval_worker is not None:
        components.append(("Eval worker", write_eval_worker_config(config, config_dir)))
    logger.info(f"Configs:\n{format_config_message(config_dir, 'pretrain', components)}")
    if config.slurm is not None:
        pretrain_slurm(config, config_path, log_dir)
        return
    if config.dry_run:
        logger.success("Dry run complete. To start the run, remove --dry-run from your command.")
        return

    logger.info(format_log_message(log_dir=log_dir, trainer=True))
    trainer_env = {**os.environ, **DEFAULT_COMMON_ENV_VARS, **DEFAULT_TRAINER_ENV_VARS, **config.env_vars}
    with ProcessGroup() as processes:
        if config.eval_worker is not None:
            # The worker's GPUs first, then the trainer's.
            eval_gpu_ids, train_gpu_ids = partition_gpus(config.eval_worker.num_gpus, config.deployment.num_train_gpus)
            trainer_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, train_gpu_ids))
            processes.start(
                "eval-worker",
                torchrun_cmd(
                    "prime_rl.trainer.pretrain.eval_worker",
                    config_dir / EVAL_WORKER_CONFIG,
                    config.eval_worker.num_gpus,
                    log_dir,
                    config.log.ranks_filter,
                ),
                env={**trainer_env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, eval_gpu_ids))},
                log_path=log_dir / "eval_worker.log",
            )
            logger.success(f"Started the eval worker on GPU(s) {' '.join(map(str, eval_gpu_ids))}")
        processes.start(
            "trainer",
            torchrun_cmd(
                "prime_rl.trainer.pretrain.train",
                config_path,
                config.deployment.num_train_gpus,
                log_dir,
                config.log.ranks_filter,
            ),
            env=trainer_env,
            log_path=log_dir / "trainer.log",
        )
        logger.success(f"Started trainer with {config.deployment.num_train_gpus} GPU(s)")
        processes.wait("trainer")
        if config.eval_worker is not None:
            # The worker scores the final snapshot after the trainer has written it.
            processes.wait("eval-worker")
        logger.success("Pretraining finished!")


def main():
    set_proc_title("Pretrain")
    pretrain(cli(PretrainConfig))


if __name__ == "__main__":
    main()
