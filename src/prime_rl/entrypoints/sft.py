from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.launcher import launch
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("SFT")
    launch(cli(SFTConfig), "sft")


if __name__ == "__main__":
    main()
