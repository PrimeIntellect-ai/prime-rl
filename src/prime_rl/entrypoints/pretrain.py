import json
import os
import uuid

from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (
    format_config_message,
    format_log_message,
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


def write_model_dir(config: PretrainConfig) -> None:
    """Write the baseline's config with ``[model.arch]`` applied, plus its tokenizer, to
    ``<run_dir>/model``, and point the model and tokenizer at it."""
    from transformers import AutoConfig, AutoTokenizer

    import prime_rl.trainer.models  # noqa: F401 - registers the custom model configs with AutoConfig

    model_dir = config.run_dir / "model"
    hf_config = AutoConfig.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    unknown = [key for key in config.model.arch if not hasattr(hf_config, key)]
    if unknown:
        raise ValueError(f"[model.arch] fields {unknown} do not exist on the {hf_config.model_type!r} config")
    for key, value in config.model.arch.items():
        setattr(hf_config, key, value)
    hf_config.save_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.name, trust_remote_code=config.tokenizer.trust_remote_code
    )
    tokenizer.save_pretrained(model_dir)
    config.model.name = config.tokenizer.name = model_dir.as_posix()


def download_data(config: PretrainConfig) -> None:
    """Download and prepare every dataset once, so the trainer ranks load them from the cache."""
    from prime_rl.trainer.pretrain.data import load_eval_documents, load_text_dataset

    if config.data.type == "text":
        load_text_dataset(config.data)
    if config.val is not None:
        for source in config.val.source:
            load_eval_documents(source.dataset, source.subset, source.split, source.revision, source.text_column)


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

        pre_download_model(config.model.name, skip_weights=config.model.debug.random_init)
        if config.model.arch:
            write_model_dir(config)
        download_data(config)

    config_path = config_dir / PRETRAIN_CONFIG
    config_path.write_text(json.dumps(dump_resolved_config(config, exclude={"dry_run", "clean"}), indent=2))
    logger.info(f"Configs:\n{format_config_message(config_dir, 'pretrain', [('Pretrain', config_path)])}")
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
