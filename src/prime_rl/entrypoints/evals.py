"""Lightweight launcher for the online evals.

Defers heavy ML imports until after ``cli()`` parses CLI args, so
``evals --help`` short-circuits in ``cli()``. The actual implementation
lives in ``prime_rl.evals.evals``.
"""

import asyncio

from prime_rl.configs.evals import EvalsConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Evals")
    config = cli(EvalsConfig)
    from prime_rl.evals.evals import run_evals

    asyncio.run(run_evals(config))


if __name__ == "__main__":
    main()
