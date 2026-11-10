from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.entrypoints.launcher import launch
from prime_rl.utils.config import check_mode, cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Pretrain")
    check_mode("pretrain")
    launch(cli(PretrainConfig), "pretrain")


if __name__ == "__main__":
    main()
