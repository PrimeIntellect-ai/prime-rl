"""Fault-injection scenario suite for multi-node RL. See docs/fault-injection.md.

Run inside a 2-node allocation. Each scenario sets PRIME_FAULT, renders the rl SLURM
script, runs it in the allocation, and checks the invariants. A run that does not
finish is resumed once, the way a requeue would resume it.

    sbatch -N 2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -t 06:00:00 \
        --wrap "uv run python scripts/fault_scenarios/run.py scripts/fault_scenarios/rl.toml outputs/faults"
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_STEPS = 12
# name: (PRIME_FAULT, seconds before a run counts as hung)
SCENARIOS = {
    "engine_crash": ("engine_crash@6", 1800),
    "broadcast_hang": ("broadcast_hang@6", 3000),
    "trainer_crash": ("trainer_crash@6", 1800),
    "orchestrator_crash": ("orchestrator_crash@6", 1800),
    "ckpt_write_fail": ("ckpt_write_fail@8", 1800),
    "env_error": ("env_error@reverse-text-b", 1800),
    "env_hang": ("env_hang@reverse-text-b", 1800),
}
# Matches every process a run leaves behind, but not this script or pgrep's own shell.
LEFTOVERS = "[v]llm|[p]rime_rl|[t]orchrun|[e]nv-server|[o]rchestrator"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300).stdout.strip()


def run(config: str, run_dir: Path, fault: str, timeout: int, extra: list[str], resume: bool) -> tuple[str, int]:
    """Render the rl script and run it in this allocation. Returns (outcome, trainer step)."""
    args = ["--resume"] if resume else ["--clean"]
    render = subprocess.run(
        [
            "rl",
            "@",
            config,
            "--output-dir",
            str(run_dir.parent),
            "--run.name",
            run_dir.name,
            "--max-steps",
            str(MAX_STEPS),
            "--dry-run",
            *args,
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if render.returncode != 0:
        raise RuntimeError(f"rl failed to render the run:\n{render.stdout}\n{render.stderr}")
    env = os.environ | {"PRIME_FAULT": fault, "PRIME_FAULT_DIR": str(run_dir / "faults")}
    log = run_dir / "launcher" / f"run_{'resume' if resume else 'first'}.log"
    with open(log, "w") as f:
        proc = subprocess.Popen(
            ["bash", str(run_dir / "launcher" / "rl.sbatch")],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            outcome = f"exit {proc.wait(timeout=timeout)}"
        except subprocess.TimeoutExpired:
            outcome = f"hung >{timeout}s"
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(60)
            os.killpg(proc.pid, signal.SIGKILL)
    job = os.environ["SLURM_JOB_ID"]
    for step in sh(f"squeue -h -s -j {job} -o %i").split():
        if not step.endswith((".batch", ".extern")):
            sh(f"scancel {step}")
    trainer_log = run_dir / "logs" / "latest" / "trainer.log"
    steps = re.findall(r"Step (\d+) \|", trainer_log.read_text(errors="ignore")) if trainer_log.exists() else []
    return outcome, max(map(int, steps), default=0)


def resumed_from(run_dir: Path) -> str:
    text = (run_dir / "logs" / "latest" / "trainer.log").read_text(errors="ignore")
    found = re.findall(r"Resuming from step (\d+)", text)
    return found[-1] if found else "-"


def leftovers() -> str:
    """Processes and GPU contexts still alive on any node after a run."""
    out = sh(
        "srun --overlap --ntasks-per-node=1 bash -c "
        f'\'echo $(hostname) procs=$(pgrep -fc "{LEFTOVERS}") '
        "gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)'"
    )
    return " ".join(line for line in out.splitlines() if not line.endswith("procs=0 gpu_procs=0")) or "none"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="multi-node rl TOML with two train sources, one named reverse-text-b")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--only", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1 :]  # passed to every rl call

    rows = []
    for name in args.only:
        fault, timeout = SCENARIOS[name]
        run_dir = args.output_dir.resolve() / name
        outcome, step = run(args.config, run_dir, fault, timeout, extra, resume=False)
        row = {"scenario": fault, "first run": f"{outcome}, step {step}", "orphans": leftovers()}
        if step >= MAX_STEPS:
            row["result"] = "finished in the first run"
        elif name.startswith("env_"):
            row["result"] = "not finished"
        else:
            outcome, step = run(args.config, run_dir, fault, timeout, extra, resume=True)
            row["resume"] = f"from step {resumed_from(run_dir)}: {outcome}, step {step}"
            row["orphans"] += f" / {leftovers()}"
            row["result"] = "finished after a resume" if step >= MAX_STEPS else "not finished"
        print(" | ".join(f"{k}: {v}" for k, v in row.items()), flush=True)
        rows.append(row)
    (args.output_dir / "summary.txt").write_text(
        "\n".join(" | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows) + "\n"
    )


if __name__ == "__main__":
    main()
