from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.launcher import launch
from prime_rl.utils.config import check_mode, cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("SFT")
    check_mode("sft")
    launch(cli(SFTConfig), "sft")


if __name__ == "__main__":
    main()
