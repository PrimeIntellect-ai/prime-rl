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
from prime_rl.utils.run_assets import configure_run_asset_env


def main():
    set_proc_title("Orchestrator")
    config = cli(OrchestratorConfig)
    configure_run_asset_env(config.output_dir, config.multimodal)
    from prime_rl.orchestrator.orchestrator import run_orchestrator

    asyncio.run(run_orchestrator(config))


if __name__ == "__main__":
    main()
