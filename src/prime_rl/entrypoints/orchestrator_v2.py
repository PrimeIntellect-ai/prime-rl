"""Lightweight launcher for orchestrator v2.

Mirrors ``prime_rl.entrypoints.orchestrator``: defers the heavy ML imports
(verifiers, transformers, pandas, etc.) until after ``cli()`` parses argv so
``orchestrator-v2 --help`` short-circuits in ~0.5 s instead of ~9 s.

The actual implementation lives in ``prime_rl.orchestrator_v2.orchestrator``.
"""

import asyncio

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("OrchestratorV2")
    config = cli(OrchestratorConfig)
    from prime_rl.orchestrator_v2.orchestrator import run_orchestrator

    asyncio.run(run_orchestrator(config))


if __name__ == "__main__":
    main()
