"""Launcher for the HF -> PrimeRL weight converter (see prime_rl.trainer.convert).

Defers the heavy ML imports until after `cli()` parses args, matching the
other entrypoints.
"""

from prime_rl.configs.convert import ConvertConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Convert")
    config = cli(ConvertConfig)
    from prime_rl.trainer.convert import run_convert

    run_convert(config)


if __name__ == "__main__":
    main()
