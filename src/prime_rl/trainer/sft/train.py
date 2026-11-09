from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.config import cli
from prime_rl.utils.pathing import check_resume_dir_mode
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("SFTTrainer")
    config = cli(SFTConfig)
    # Fail before the trainer's heavy imports when resume.dir forks another run kind's checkpoint.
    check_resume_dir_mode(config.resume, "sft", ckpt=config.ckpt)
    from prime_rl.trainer.train import train

    train(config)


if __name__ == "__main__":
    main()
