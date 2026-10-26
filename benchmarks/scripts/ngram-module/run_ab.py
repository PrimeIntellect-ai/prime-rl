"""Launch one run of the n-gram module A/B from an arm's frozen worktree and record it in run.json."""

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
BASE_CONFIG = HARNESS_DIR / "configs" / "ab.toml"
SCRATCH = Path("/home/garrett/tmp/ngram-module")
DEFAULT_ROOT = SCRATCH / "ab"
CONVERSION_ROOT = SCRATCH / "conv"
TRITON_ROOT = Path("/tmp") / os.environ.get("USER", "user") / "ngram-module-triton"
MAX_TRACE_STEPS = 9
DEFAULT_TRACE_STEPS = 5
TAIL_LINES = 80
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EXCEPTION_LINE = re.compile(r"^(?:\[[^\]]*\]:)*\s*(?:[\w.]+\.)?\w*(?:Error|Exception|Interrupt)\b.*")


@dataclass(frozen=True)
class Cell:
    gpus: int
    ac: bool
    reshard: bool
    ep: int
    cp: int


CELLS = {
    1: Cell(gpus=1, ac=False, reshard=True, ep=1, cp=1),
    2: Cell(gpus=1, ac=True, reshard=True, ep=1, cp=1),
    3: Cell(gpus=8, ac=False, reshard=True, ep=8, cp=1),
    4: Cell(gpus=8, ac=False, reshard=False, ep=8, cp=1),
    5: Cell(gpus=8, ac=True, reshard=True, ep=8, cp=1),
    6: Cell(gpus=8, ac=True, reshard=False, ep=8, cp=1),
    7: Cell(gpus=8, ac=True, reshard=True, ep=1, cp=1),
    8: Cell(gpus=8, ac=True, reshard=True, ep=8, cp=2),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", help="arm name, e.g. main or branch; also names the arm's conversion dir")
    parser.add_argument("worktree", type=Path, help="the arm's frozen worktree")
    parser.add_argument("cell", type=int, choices=sorted(CELLS), help="matrix cell")
    parser.add_argument("repeat", type=int, help="repeat index")
    parser.add_argument("--steps", type=int, default=None, help="max_steps (default: ab.toml, or 5 with --trace)")
    parser.add_argument("--gpus", default=None, help="device list for CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3")
    parser.add_argument("--trace", action="store_true", help="write a torch.profiler trace per rank")
    parser.add_argument("--graph-breaks", action="store_true", help="add graph_breaks to TORCH_LOGS")
    parser.add_argument("--venv", type=Path, default=None, help="node-local venv from make_local_venv.sh")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="CLI overrides appended last, shell-split; write --extra='--model.ep 4'",
    )
    parser.add_argument("--tag", default=None, help="suffix for the cell directory, for --extra variants")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="output root")
    parser.add_argument("--dry-run", action="store_true", help="print the command and environment, run nothing")
    return parser.parse_args()


def git(worktree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(worktree), *args], check=True, capture_output=True, text=True).stdout


def cell_dir_name(cell_id: int, num_gpus: int, args) -> str:
    parts = [f"cell{cell_id}"]
    if num_gpus != CELLS[cell_id].gpus:
        parts.append(f"{num_gpus}gpu")
    if args.trace:
        parts.append("trace")
    if args.graph_breaks:
        parts.append("graph-breaks")
    if args.tag:
        parts.append(args.tag)
    return "-".join(parts)


def cell_overrides(cell: Cell, num_gpus: int) -> list[str]:
    return [
        "--deployment.num-train-gpus",
        str(num_gpus),
        *(["--model.ac.mode", "full"] if cell.ac else ["--model.ac", "None"]),
        "--model.reshard-after-forward",
        "true" if cell.reshard else "false",
        "--model.ep",
        str(cell.ep),
        "--model.cp",
        str(cell.cp),
    ]


def tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    return [ANSI.sub("", line) for line in path.read_text(errors="replace").splitlines()[-lines:]]


def rank_logs(run_dir: Path) -> dict[str, dict[str, str]]:
    logs: dict[str, dict[str, str]] = {}
    for path in sorted(run_dir.glob("logs/attempt_*/trainer/torchrun/*/attempt_*/*/*.log")):
        logs.setdefault(path.parent.name, {})[path.stem] = str(path)
    return dict(sorted(logs.items(), key=lambda item: int(item[0])))


def last_exception(paths: list[Path]) -> str | None:
    matches = [line for path in paths for line in tail(path, 400) if EXCEPTION_LINE.match(line)]
    return matches[-1].strip() if matches else None


def traceback_tail(paths: list[Path]) -> list[str]:
    for path in paths:
        lines = tail(path, 2000)
        starts = [i for i, line in enumerate(lines) if "Traceback (most recent call last)" in line]
        if starts:
            return lines[starts[-1] : starts[-1] + TAIL_LINES]
    return []


def write_json(path: Path, record: dict) -> None:
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(record, indent=2) + "\n")
    staging.replace(path)


def main():
    args = parse_args()
    worktree = args.worktree.resolve()
    cell = CELLS[args.cell]
    devices = args.gpus.split(",") if args.gpus else None
    num_gpus = len(devices) if devices else cell.gpus
    steps = args.steps if args.steps is not None else (DEFAULT_TRACE_STEPS if args.trace else None)
    if args.trace and (steps is None or steps > MAX_TRACE_STEPS):
        raise SystemExit(f"--trace needs --steps <= {MAX_TRACE_STEPS} (the trainer rejects longer traces)")

    cell_dir = cell_dir_name(args.cell, num_gpus, args)
    out_dir = args.root.resolve() / args.arm / cell_dir / f"r{args.repeat}"
    run_dir = out_dir / "ab"
    conversion_dir = CONVERSION_ROOT / args.arm
    head = git(worktree, "rev-parse", "HEAD").strip()
    dirty = git(worktree, "status", "--porcelain").splitlines()

    triton_dir = TRITON_ROOT / args.arm / cell_dir / f"r{args.repeat}-<fresh>"
    if not args.dry_run:
        if args.venv:
            venv_worktree = (args.venv / "prime-rl-worktree").read_text().strip()
            if venv_worktree != str(worktree):
                raise SystemExit(f"{args.venv} was built for {venv_worktree}, not {worktree}")
        if dirty:
            raise SystemExit(f"refusing to run a dirty worktree {worktree}:\n" + "\n".join(dirty))
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite {out_dir}; delete it to rerun")
        out_dir.mkdir(parents=True, exist_ok=True)
        (TRITON_ROOT / args.arm / cell_dir).mkdir(parents=True, exist_ok=True)
        triton_dir = Path(tempfile.mkdtemp(prefix=f"r{args.repeat}-", dir=TRITON_ROOT / args.arm / cell_dir))

    command = [
        "uv",
        "run",
        *(["--active"] if args.venv else []),
        "--no-sync",
        "sft",
        "@",
        str(BASE_CONFIG),
        "--model.conversion-dir",
        str(conversion_dir),
        *cell_overrides(cell, num_gpus),
        *(["--max-steps", str(steps)] if steps is not None else []),
        *(["--trace-path", str(out_dir / "trace")] if args.trace else []),
        "--env-vars",
        json.dumps({"TRITON_CACHE_DIR": str(triton_dir)}),
        *[token for extra in args.extra for token in shlex.split(extra)],
    ]
    env_overrides = {
        "PRL_OUTPUT_DIR": str(out_dir),
        "TORCHINDUCTOR_CACHE_DIR": str(out_dir / "inductor"),
        "TORCH_LOGS": "recompiles,graph_breaks" if args.graph_breaks else "recompiles",
    }
    if devices:
        env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.venv:
        env_overrides["VIRTUAL_ENV"] = str(args.venv.resolve())

    if args.dry_run:
        print(f"out_dir: {out_dir}")
        print(
            f"worktree: {worktree} at {head}" + (f" (DIRTY: {len(dirty)} entries, a real run refuses)" if dirty else "")
        )
        print("env: " + " ".join(f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()))
        print(f"cwd: {worktree}")
        print(shlex.join(command))
        return

    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"} | env_overrides
    launcher_log = out_dir / "launcher.log"
    record = {
        "arm": args.arm,
        "worktree": str(worktree),
        "head": head,
        "harness_head": git(HARNESS_DIR, "rev-parse", "HEAD").strip(),
        "harness_dirty": bool(git(HARNESS_DIR, "status", "--porcelain", "--", str(HARNESS_DIR)).strip()),
        "cell": args.cell,
        "cell_dir": cell_dir,
        "cell_config": asdict(cell),
        "num_gpus": num_gpus,
        "repeat": args.repeat,
        "steps": steps,
        "trace": args.trace,
        "graph_breaks": args.graph_breaks,
        "venv": str(args.venv.resolve()) if args.venv else None,
        "extra": args.extra,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cwd": str(worktree),
        "command": shlex.join(command),
        "env_overrides": env_overrides,
        "triton_cache_dir": str(triton_dir),
        "out_dir": str(out_dir),
        "run_dir": str(run_dir),
        "launcher_log": str(launcher_log),
        "trainer_log": str(run_dir / "logs" / "attempt_1" / "trainer.log"),
        "metrics": str(run_dir / "monitors" / "file" / "metrics.jsonl"),
        "trace_dir": str(out_dir / "trace") if args.trace else None,
        "start_time": time.time(),
        "end_time": None,
        "exit_code": None,
    }
    write_json(out_dir / "run.json", record)
    print(f"{args.arm} {cell_dir} r{args.repeat}: {out_dir}", flush=True)

    with launcher_log.open("w") as log:
        exit_code = subprocess.run(command, cwd=worktree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    logs = rank_logs(run_dir)
    record |= {
        "end_time": time.time(),
        "exit_code": exit_code,
        "rank_logs": logs,
    }
    if exit_code != 0:
        record["rank_errors"] = {
            rank: last_exception([Path(f) for f in files.values()]) for rank, files in logs.items()
        }
        failed = [
            Path(files[name])
            for rank, files in logs.items()
            if record["rank_errors"][rank]
            for name in sorted(files, reverse=True)
        ]
        record["error_tail"] = traceback_tail(failed) or tail(launcher_log, TAIL_LINES)
    write_json(out_dir / "run.json", record)
    print(f"exit {exit_code} after {record['end_time'] - record['start_time']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
