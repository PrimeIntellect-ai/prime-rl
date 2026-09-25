"""Launch a configured Flow: `flow @ run.toml`; inspect, apply and drain control its boundaries."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from verifiers.v1.configs.flow import FlowConfig as PipelineConfig
from verifiers.v1.flow import Flow, Transition, drain_on_interrupt
from verifiers.v1.flow.flow import DRAIN_FILE, JOBS, TRANSITIONS, job_path
from verifiers.v1.flow.job import STATE, Job, JobInspection
from verifiers.v1.utils.loaders import load_flow

from prime_rl.configs.flow import ApplyConfig, DrainConfig, FlowConfig, InspectConfig
from prime_rl.utils.config import cli, dump_resolved_config


def launch(config: FlowConfig) -> int:
    root = config.run_dir
    from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
    from prime_rl.utils.logger import InterceptHandler, setup_logger
    from prime_rl.utils.pathing import prepare_attempt_dirs, write_launch_artifacts

    config_dir, log_dir = prepare_attempt_dirs(root)
    write_launch_artifacts(config_dir, "flow")
    (config_dir / "flow.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    log_file = log_dir / "flow.log"
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file)
    logging.basicConfig(level=logging.INFO, handlers=[InterceptHandler(prefix=None)], force=True)
    logging.getLogger("verifiers").setLevel(config.log.vf_level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if config.dashboard:
        log_dashboard_url(logger, ensure_dashboard(root.parent, logger))
    setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file, console_level="ERROR")

    async def run() -> int:
        flow = load_flow(config.flow, root=root)
        with drain_on_interrupt(flow):
            result = await flow.run()
        setup_logger(config.log.level, json_logging=config.log.json_logging, log_file=log_file).info(
            "Flow finished: {} {}", result.reason, result.counts
        )
        return flow.exit_code(result)

    return asyncio.run(run())


class Inspection(BaseModel):
    jobs: list[JobInspection[Any]]
    events: Path
    traces: Path
    calls: Path
    draining: bool


def inspect(root: Path, name: str | None = None) -> Inspection:
    root = root.resolve()
    paths = [job_path(root, name)] if name else sorted((root / JOBS).iterdir())
    return Inspection(
        jobs=[Job(path).inspect() for path in paths if (path / STATE).is_file()],
        events=root / TRANSITIONS,
        traces=root / "traces.jsonl",
        calls=root / "calls",
        draining=(root / DRAIN_FILE).exists(),
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args and args[0] in ("inspect", "apply", "drain") else "run"
    if command == "run":
        raise SystemExit(launch(cli(FlowConfig, args=args, prog="flow")))
    if command == "inspect":
        config = cli(InspectConfig, args=args, prog="flow inspect")
        print(inspect(config.root, config.job).model_dump_json(indent=2))
    elif command == "drain":
        config = cli(DrainConfig, args=args, prog="flow drain")
        (config.root / DRAIN_FILE).touch()
    else:
        config = cli(ApplyConfig, args=args, prog="flow apply")
        flow = Flow(PipelineConfig(), root=config.root)
        job = flow.job(config.job)
        data = job.data_type.model_validate_json(config.data_file.read_text()) if config.data_file else None
        revision = flow.apply(
            job.id,
            Transition(
                stage=config.stage,
                status=config.status,
                reason=config.reason,
                note=config.note,
                outcome=config.outcome,
                report=config.report,
                data=data,
            ),
            expected=config.expected,
        )
        print(json.dumps({"revision": revision}))


if __name__ == "__main__":
    main()
