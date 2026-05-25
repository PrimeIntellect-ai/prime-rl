"""Lightweight launcher for the orchestrator.

Defers heavy ML imports (verifiers, transformers, pandas, prime_rl.trainer.*)
until after ``cli()`` parses CLI args, so ``orchestrator --help`` short-circuits
in ``cli()`` and returns in ~0.5 s instead of ~9 s.

The actual orchestrator implementation lives in
``prime_rl.orchestrator.orchestrator``, which is also runnable as
``python -m prime_rl.orchestrator.orchestrator``.
"""

import asyncio

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Orchestrator")
    config = cli(OrchestratorConfig)
    from prime_rl.orchestrator.orchestrator import orchestrate

    asyncio.run(orchestrate(config))


if __name__ == "__main__":
    main()
