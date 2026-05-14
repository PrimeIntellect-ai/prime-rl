from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.controller import run_sweep
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Sweep")
    run_sweep(cli(SweepConfig))


if __name__ == "__main__":
    main()
