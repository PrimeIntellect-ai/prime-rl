from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.trainer.train import train
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("PretrainTrainer")
    train(cli(PretrainConfig))


if __name__ == "__main__":
    main()
