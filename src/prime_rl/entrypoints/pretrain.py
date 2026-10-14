import json
import os
import uuid
from pathlib import Path

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
