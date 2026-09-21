"""Launch, inspect and steer pipelines: `uv run flow {run,inspect,steer,drain} --help`."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, TypeVar, get_args

from pydantic import BaseModel
from verifiers.v1.flow import FlowConfig, FlowEntrypoint
from verifiers.v1.flow.events import Status
from verifiers.v1.flow.flow import DRAIN_FILE, TRANSITIONS, UNITS, unit_path
from verifiers.v1.flow.unit import Unit, UnitInspection

from prime_rl.utils.config import cli, dump_resolved_config

C = TypeVar("C", bound=FlowConfig)


def launch(entrypoint: FlowEntrypoint[C], root: Path, args: list[str], *, dashboard: bool) -> int:
    config = cli(entrypoint.config_type, args=args, prog="flow run")

    from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
    from prime_rl.utils.logger import InterceptHandler, setup_logger
    from prime_rl.utils.pathing import prepare_attempt_dirs, write_launch_artifacts

    config_dir, log_dir = prepare_attempt_dirs(root)
    write_launch_artifacts(config_dir, "flow")
    (config_dir / "flow.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    log_file = log_dir / "flow.log"
    logger = setup_logger(log_file=log_file)
    logging.basicConfig(level=logging.INFO, handlers=[InterceptHandler(prefix=None)], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if dashboard:
        log_dashboard_url(logger, ensure_dashboard(root.parent, logger))
    setup_logger(log_file=log_file, console_level="ERROR")
    return asyncio.run(entrypoint.run(root, config))


class Inspection(BaseModel):
    units: list[UnitInspection[Any]]
    events: Path
    traces: Path
    calls: Path
    draining: bool


def inspect(root: Path, name: str | None = None) -> Inspection:
    root = root.resolve()
    paths = [unit_path(root, name)] if name else sorted((root / UNITS).iterdir())
    return Inspection(
        units=[Unit(path).inspect() for path in paths if (path / ".git").exists()],
        events=root / TRANSITIONS,
        traces=root / "traces.jsonl",
        calls=root / "calls",
        draining=(root / DRAIN_FILE).exists(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Launch an installed FlowEntrypoint")
    run.add_argument("--no-dashboard", action="store_true", help="Skip dashboard registration and startup")
    run.add_argument("entrypoint", help="Dotted import path, e.g. repoforge.launch.entrypoint")
    run.add_argument("root", type=Path)
    run.add_argument("config_args", nargs=argparse.REMAINDER, help="@ config.toml and typed pipeline overrides")
    view = commands.add_parser("inspect", help="Committed states and executing stages as JSON")
    view.add_argument("root", type=Path)
    view.add_argument("unit", nargs="?")
    steer = commands.add_parser("steer", help="Publish boundary controls or settled data updates")
    steer.add_argument("root", type=Path)
    steer.add_argument("unit")
    steer.add_argument("--stage")
    steer.add_argument("--status", choices=get_args(Status))
    steer.add_argument("--reason")
    steer.add_argument("--note")
    steer.add_argument("--data", type=Path, help="JSON object of pipeline data fields to update")
    steer.add_argument("--expected", help="Workflow revision; required for data updates")
    drain = commands.add_parser("drain", help="Finish running calls and stop admission")
    drain.add_argument("root", type=Path)
    options = parser.parse_args(argv)
    if options.command == "run":
        from prime_rl.utils.utils import import_object

        entrypoint = import_object(options.entrypoint)
        if not isinstance(entrypoint, FlowEntrypoint):
            raise TypeError(f"{options.entrypoint} must be a FlowEntrypoint")
        raise SystemExit(launch(entrypoint, options.root, options.config_args, dashboard=not options.no_dashboard))
    elif options.command == "inspect":
        print(inspect(options.root, options.unit).model_dump_json(indent=2))
    elif options.command == "drain":
        (options.root / DRAIN_FILE).touch()
    else:
        unit = Unit(unit_path(options.root, options.unit))
        revision = unit.steer(
            stage=options.stage,
            status=options.status,
            reason=options.reason,
            note=options.note,
            expected=options.expected,
            data=json.loads(options.data.read_text()) if options.data else None,
        )
        print(json.dumps({"revision": revision}))


if __name__ == "__main__":
    main()
