"""Lightweight launcher for the online evals.

Defers heavy ML imports until after ``cli()`` parses CLI args, so
``evals --help`` short-circuits in ``cli()``. The actual implementation
lives in ``prime_rl.evals.evals``.
"""

import asyncio
import json
import os

from prime_rl.configs.evals import EvalsConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Evals")
    config = cli(EvalsConfig)
    from prime_rl.utils.pathing import prepare_attempt_dirs, write_launch_toml

    config_dir, log_dir = prepare_attempt_dirs(config.output_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    os.environ["PRL_LOG_DIR"] = str(log_dir)
    write_launch_toml(config_dir, "evals")
    (config_dir / "evals.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    from prime_rl.evals.evals import run_evals

    asyncio.run(run_evals(config))


if __name__ == "__main__":
    main()
