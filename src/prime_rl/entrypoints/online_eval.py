"""Launcher for ``online-eval``: evaluate the trainer's weight broadcasts as they appear.

Defers heavy ML imports until after ``cli()`` parses CLI args, so
``online-eval --help`` short-circuits in ``cli()``. The implementation lives in
``prime_rl.eval.online``.
"""

import asyncio
import json
import os

from prime_rl.configs.eval import OnlineEvalConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("OnlineEvals")
    config = cli(OnlineEvalConfig)
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.pathing import prepare_attempt_dirs, write_launch_artifacts

    config_dir, log_dir = prepare_attempt_dirs(config.output_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    write_launch_artifacts(config_dir, "online-eval")
    (config_dir / "online_eval.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    from prime_rl.eval.online import run_online_eval

    asyncio.run(run_online_eval(config, log_dir))


if __name__ == "__main__":
    main()
