import os

# The torchrun agent, read before the slow imports below (see die_with_parent in main)
PARENT_PID = os.getppid()

from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.trainer.train import train
from prime_rl.utils.config import cli
from prime_rl.utils.process import die_with_parent, set_proc_title


def main():
    set_proc_title("PretrainTrainer")
    # torchrun starts its workers in their own sessions; tie them to the agent
    die_with_parent(PARENT_PID)
    train(cli(PretrainConfig))


if __name__ == "__main__":
    main()
