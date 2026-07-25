import os

from prime_rl.configs.reward_model import RewardModelConfig
from prime_rl.entrypoints.local_trainer import launch_local_trainer
from prime_rl.utils.config import cli
from prime_rl.utils.pathing import validate_output_dir
from prime_rl.utils.process import set_proc_title


def reward_model(config: RewardModelConfig) -> None:
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    validate_output_dir(config.output_dir, resuming=False, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.model.name)

    launch_local_trainer(
        config,
        config_filename="reward_model.toml",
        trainer_module="prime_rl.trainer.reward_model.train",
        display_name="reward-model",
    )


def main() -> None:
    set_proc_title("RewardModel")
    reward_model(cli(RewardModelConfig))


if __name__ == "__main__":
    main()
