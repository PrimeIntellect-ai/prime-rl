"""Lightweight launcher for the TTT service.

Defers the heavy imports (torch, transformers, peft) until after ``cli()`` parses CLI
args, so ``ttt --help`` short-circuits quickly.
"""

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("TTT")
    config = cli(TTTServiceConfig)
    from prime_rl.utils.logger import setup_logger

    setup_logger(config.log.level, json_logging=config.log.json_logging)
    from prime_rl.ttt.server import run_server

    run_server(config)


if __name__ == "__main__":
    main()
