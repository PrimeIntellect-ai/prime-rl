"""Launcher for ``uv run eval``: one epoch of every configured eval source.

Defers heavy ML imports until after the config is parsed, so ``eval --help``
short-circuits. The implementation lives in ``prime_rl.eval.eval``.
"""

import asyncio
import json
import os
import re
import signal
import sys
import tomllib
import uuid
from pathlib import Path
from subprocess import Popen

from verifiers.v1.cli.resolve import narrow_config, plugin_errors, with_positional_taskset

from prime_rl.configs.eval import EvalConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, cleanup_processes, set_proc_title

USAGE = """\
usage: uv run eval [<taskset-id>] [--env.<field> <value> ...] [-n N] [-r N] [-c N] [-m MODEL] [options]
       uv run eval @ eval.toml [options]                                  multi-source runs ([[source]] blocks)
       uv run eval @ eval.toml --run.name <name> --resume                 resume an interrupted run

Shorthands:
  <taskset-id>             the taskset of the run's only source (--env.taskset.id)
  --env.<field> <value>    a field of the env block every source inherits (e.g. --env.agent.harness.id bash)
  -c N                     pin the concurrency band (concurrency.min_inflight = max_inflight = N)
"""

NUMBER = re.compile(r"-?\d+(\.\d+)?")


def expand_shorthands(argv: list[str]) -> list[str]:
    """Rewrite the shorthands into flags ``EvalConfig`` parses.

    ``<taskset-id>`` becomes ``--env.taskset.id``; a run without ``[[source]]`` blocks
    evaluates the env block as its only source. ``-c N`` pins the concurrency band.
    Everything else, ``--env.*`` included, passes through untouched.
    """
    if argv and not argv[0].startswith(("-", "@")) and any(toml_defines_source(p) for p in root_config_files(argv)):
        raise SystemExit(
            "The <taskset-id> shorthand names the run's only source and cannot be combined "
            "with a config file that defines [[source]] blocks - use one or the other"
        )
    rest = with_positional_taskset(argv)
    out: list[str] = []
    i = 0
    while i < len(rest):
        flag, has_value, value = rest[i].partition("=")
        if flag != "-c":
            out.append(rest[i])
        else:
            if not has_value:
                if i + 1 >= len(rest) or (rest[i + 1].startswith("-") and not NUMBER.fullmatch(rest[i + 1])):
                    raise SystemExit("-c needs a value")
                i += 1
                value = rest[i]
            out += ["--concurrency.min_inflight", value, "--concurrency.max_inflight", value]
        i += 1
    return out


def root_config_files(argv: list[str]) -> list[Path]:
    """Root ``@ file`` references (a ``--flag @ file`` is a nested reference)."""
    return [
        Path(argv[i + 1])
        for i, arg in enumerate(argv[:-1])
        if arg == "@" and (i == 0 or not argv[i - 1].startswith("--"))
    ]


def toml_defines_source(path: Path) -> bool:
    if path.suffix != ".toml" or not path.is_file():
        return False
    with path.open("rb") as f:
        return "source" in tomllib.load(f)


def main():
    set_proc_title("Eval")
    argv = sys.argv[1:]
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(narrow_config(EvalConfig, with_positional_taskset(argv)))
        return
    # The typed parse sees the expanded flags; the launch artifacts keep the command as typed.
    expanded = expand_shorthands(argv)
    sys.argv = [sys.argv[0], *expanded]
    with plugin_errors():
        config = cli(narrow_config(EvalConfig, expanded))
    sys.argv = [sys.argv[0], *argv]

    from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.pathing import (
        format_config_message,
        format_log_message,
        prepare_attempt_dirs,
        validate_run_dir,
        write_env_server_config,
        write_launch_artifacts,
    )

    # The run identity is runtime-only: $PRL_RUN_ID / $PRL_RUN_NAME are stamped on
    # every episode and inherited by the env servers.
    os.environ.setdefault("PRL_RUN_ID", uuid.uuid4().hex)
    assert config.run.name is not None  # resolved at construction
    os.environ["PRL_RUN_NAME"] = config.run.name

    clean = config.clean and not os.environ.get("NEVER_CLEAN")
    validate_run_dir(config.run_dir, output_dir=config.output_dir, resuming=config.resume, clean=clean)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    log_file = log_dir / "eval.log"
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file)
    logger.info("Starting eval")

    write_launch_artifacts(config_dir, "eval")
    (config_dir / "eval.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    components: list[tuple[str, Path | str]] = [("Eval", config_dir / "eval.json")]
    # One env server per source without an explicit `serve.address`, like the rl launcher's:
    # `env-server @ <path>` binds an OS-assigned port and publishes it to the source's
    # address file, where the eval picks it up.
    env_servers = [source for source in config.source if source.serve.address is None]
    env_names = [source.resolved_name for source in env_servers]
    if env_servers:
        components.append(("Envs", f"{config_dir}/envs/eval/*.json"))
        for source in env_servers:
            components.append(
                (f" {source.resolved_name}", write_env_server_config(config_dir, "eval", source, config.log))
            )
    if config.dry_run:
        logger.info(f"Configs:\n{format_config_message(config_dir, 'eval', components)}")
        logger.success("Dry run complete. To start the eval, remove --dry-run from your command.")
        return

    dashboard_url = ensure_dashboard(config.output_dir, logger) if config.dashboard else None
    from prime_rl.eval.eval import run_eval

    processes: list[Popen] = []
    for source in env_servers:
        name = source.resolved_name
        logger.info(f"Starting {name} server")
        env_server_log = log_dir / "envs" / "eval" / f"{name}.log"
        env_server_log.parent.mkdir(parents=True, exist_ok=True)
        with open(env_server_log, "w") as log_file_handle:
            processes.append(
                Popen(
                    ["env-server", "@", (config_dir / "envs" / "eval" / f"{name}.json").as_posix()],
                    env={**os.environ, **DEFAULT_COMMON_ENV_VARS},
                    stdout=log_file_handle,
                    stderr=log_file_handle,
                )
            )

    logger.info(f"Configs:\n{format_config_message(config_dir, 'eval', components)}")
    logger.info(format_log_message(log_dir, eval=True, env_names={"eval": env_names}))

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)
    log_dashboard_url(logger, dashboard_url)

    # Like the rl/sft launchers, the console stays quiet while the eval runs: results
    # live in the dashboard and the log file, only errors surface here.
    setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file, console_level="ERROR")
    try:
        asyncio.run(run_eval(config))
    finally:
        cleanup_processes(processes)
    setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file).success("Eval finished!")


if __name__ == "__main__":
    main()
