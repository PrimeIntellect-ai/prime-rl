"""Lightweight launcher for the online evaluator.

Defers heavy ML imports until after ``cli()`` parses CLI args, so
``evaluator --help`` short-circuits in ``cli()``. The actual implementation
lives in ``prime_rl.orchestrator.evaluator``.
"""

import asyncio

from prime_rl.configs.evaluator import EvaluatorConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Evaluator")
    config = cli(EvaluatorConfig)
    from prime_rl.orchestrator.evaluator import run_evaluator

    asyncio.run(run_evaluator(config))


if __name__ == "__main__":
    main()
