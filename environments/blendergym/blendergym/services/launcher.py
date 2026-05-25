"""Config-driven service launcher.

Short-lived process: reads services.toml, starts each service as a
subprocess, runs a 3-second smoke check, then prints shell-eval-able
export statements and exits.  NOT a supervisor — runtime monitoring
is handled by client-side diagnose_service_down().
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .health import clear_sentinels

DEFAULT_CONFIG = str(
    importlib.resources.files("blendergym.services") / "services.toml"
)

_ENV_REF_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


@dataclass
class ServiceSpec:
    name: str
    module: str
    port: int
    health_endpoint: str = "/health"
    extra_args: list[str] = field(default_factory=list)


def load_services(config_path: str) -> list[ServiceSpec]:
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return [ServiceSpec(**svc) for svc in data["services"]]


def _validate_env_refs(args: list[str], spec_name: str) -> None:
    """Verify every $VAR / ${VAR} reference in extra_args is exported."""
    for arg in args:
        for match in _ENV_REF_RE.finditer(arg):
            var = match.group(1) or match.group(2)
            if not os.environ.get(var):
                print(
                    f"FATAL: {spec_name} requires env var ${var} "
                    f"in arg: {arg!r}\n"
                    f"Did you forget to 'export' the variable?",
                    file=sys.stderr,
                )
                sys.exit(1)


def _expand_env(args: list[str]) -> list[str]:
    return [os.path.expandvars(a) for a in args]


def _make_log_config(service_name: str, log_dir: str) -> dict:
    """Generate a uvicorn JSON log config that writes to $LOG_DIR/{name}.log."""
    log_file = str(Path(log_dir) / f"{service_name}.log")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "json",
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "json",
                "level": "WARNING",
            },
        },
        "root": {
            "handlers": ["file", "stderr"],
            "level": "INFO",
        },
    }


def _start(
    spec: ServiceSpec, log_dir: str
) -> tuple[subprocess.Popen, Path]:
    clear_sentinels(spec.name)

    log_cfg = _make_log_config(spec.name, log_dir)
    cfg_path = Path(log_dir) / f"{spec.name}_log_config.json"
    cfg_path.write_text(json.dumps(log_cfg))

    diag_log = Path(log_dir) / f"{spec.name}_diag.log"

    _validate_env_refs(spec.extra_args, spec.name)
    expanded_args = _expand_env(spec.extra_args)

    cmd = [
        sys.executable,
        "-m",
        spec.module,
        "--port",
        str(spec.port),
        "--log-config",
        str(cfg_path),
        *expanded_args,
    ]
    diag_file = open(diag_log, "w")
    proc = subprocess.Popen(cmd, stdout=diag_file, stderr=diag_file)
    diag_file.close()
    return proc, diag_log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch BlenderGym sidecar services"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Path to services.toml"
    )
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args(argv)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    specs = load_services(args.config)
    procs: dict[str, tuple[subprocess.Popen, Path]] = {}
    for spec in specs:
        procs[spec.name] = _start(spec, args.log_dir)

    time.sleep(3)
    for name, (proc, diag_log) in procs.items():
        if proc.poll() is not None:
            content = diag_log.read_text() if diag_log.exists() else "(empty)"
            print(
                f"FATAL: {name} crashed (exit={proc.returncode})\n{content}",
                file=sys.stderr,
            )
            sys.exit(1)

    parts: list[str] = []
    all_pids: list[str] = []
    for name, (proc, diag_log) in procs.items():
        prefix = name.upper()
        parts.append(f"export {prefix}_PID={proc.pid}")
        parts.append(
            f"export {prefix}_STDERR_LOG={shlex.quote(str(diag_log))}"
        )
        all_pids.append(str(proc.pid))
    parts.append(f"export SVC_PIDS={shlex.quote(' '.join(all_pids))}")
    print("; ".join(parts))


if __name__ == "__main__":
    main()
