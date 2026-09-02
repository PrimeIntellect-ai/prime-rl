"""Launcher for ``uv run eval``: one epoch of every configured eval source.

Defers heavy ML imports until after the config is parsed, so ``eval --help``
short-circuits. The implementation lives in ``prime_rl.eval.eval``.
"""

import asyncio
import json
import os
import re
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any

from prime_rl.configs.eval import EvalConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.process import set_proc_title

USAGE = """\
usage: uv run eval [<taskset-id>] [--env.<field> <value> ...] [-n N] [-r N] [-c N] [-m MODEL] [options]
       uv run eval @ eval.toml [options]                                  multi-source runs ([[source]] blocks)
       uv run eval @ eval.toml --run.name <name> --resume                 resume an interrupted run

Shorthands for a single-source run:
  <taskset-id>             the taskset of the run's only source
  --env.<field> <value>    a field of that source's env block (e.g. --env.agent.harness.id bash)
  -c N                     pin the concurrency band (concurrency.min_inflight = max_inflight = N)
"""

NUMBER = re.compile(r"-?\d+(\.\d+)?")


def set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def parse_value(raw: str) -> Any:
    """A JSON list/object stays structured; every other value is left to pydantic."""
    return json.loads(raw) if raw.startswith(("[", "{")) else raw


def expand_shorthands(argv: list[str]) -> list[str]:
    """Rewrite the single-source shorthands into flags ``EvalConfig`` parses.

    ``<taskset-id>`` and ``--env.<path> <value>`` describe the run's only source and fold
    into one JSON ``--source`` flag (pydantic-config has no list-index paths, so
    ``--source.0.env...`` cannot address it). ``-c N`` pins the concurrency band.
    Everything else passes through untouched.
    """
    out: list[str] = []
    source: dict[str, Any] = {}
    rest = list(argv)
    if rest and not rest[0].startswith(("-", "@")):
        set_nested(source, ["env", "taskset", "id"], rest.pop(0))
    i = 0
    while i < len(rest):
        arg = rest[i]
        flag, has_value, value = arg.partition("=")
        if not has_value and (flag.startswith("--env.") or flag == "-c"):
            if i + 1 >= len(rest) or (rest[i + 1].startswith("-") and not NUMBER.fullmatch(rest[i + 1])):
                raise SystemExit(f"{flag} needs a value")
            i += 1
            value = rest[i]
        if flag.startswith("--env."):
            set_nested(source, [key.replace("-", "_") for key in flag[2:].split(".")], parse_value(value))
        elif flag == "-c":
            out += ["--concurrency.min_inflight", value, "--concurrency.max_inflight", value]
        else:
            out.append(arg)
        i += 1
    if source:
        if any(toml_defines_source(path) for path in root_config_files(argv)):
            raise SystemExit(
                "The <taskset-id> / --env.* shorthands describe a single source and cannot be combined "
                "with a config file that defines [[source]] blocks - use one or the other"
            )
        out += ["--source", json.dumps([source])]
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
        cli(EvalConfig)
        return
    # The typed parse sees the expanded flags; the launch artifacts keep the command as typed.
    sys.argv = [sys.argv[0], *expand_shorthands(argv)]
    config = cli(EvalConfig)
    sys.argv = [sys.argv[0], *argv]

    from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.pathing import prepare_attempt_dirs, validate_run_dir, write_launch_artifacts

    # The run identity is runtime-only: $PRL_RUN_ID / $PRL_RUN_NAME are stamped on
    # every episode and inherited by the env servers.
    os.environ.setdefault("PRL_RUN_ID", uuid.uuid4().hex)
    assert config.run.name is not None  # resolved at construction
    os.environ["PRL_RUN_NAME"] = config.run.name

    clean = config.clean and not os.environ.get("NEVER_CLEAN")
    validate_run_dir(config.run_dir, output_dir=config.output_dir, resuming=config.resume is not None, clean=clean)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    write_launch_artifacts(config_dir, "eval")
    (config_dir / "eval.json").write_text(json.dumps(dump_resolved_config(config), indent=2))

    log_file = log_dir / "eval.log"
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file)
    logger.info(f"Wrote config to {config_dir}")
    if config.dry_run:
        logger.success("Dry run complete. To start the eval, remove --dry-run from your command.")
        return

    names = ", ".join(source.resolved_name for source in config.source)
    logger.info(f"Starting eval of {names} with {config.model} ({config.client.base_url})")
    logger.info(f"Logs:\n  {'Eval:':<18}tail -F {log_file}\n  {'Envs:':<18}tail -F {log_dir}/envs/eval/*.log")
    dashboard_url = ensure_dashboard(config.output_dir, logger) if config.dashboard else None
    log_dashboard_url(logger, dashboard_url)
    from prime_rl.eval.eval import run_eval

    # The console shows results and problems from here on; the log file keeps everything.
    setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file, console_level="SUCCESS")
    asyncio.run(run_eval(config, log_dir))


if __name__ == "__main__":
    main()
