"""Benchmark one arm of the fp32-router compile comparison. Run it from the arm's worktree on one 8-GPU node."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

NUM_GPUS = 8
EP = 8
SEQ_LEN = 16384
BATCH_SIZE = 16
MAX_STEPS = 20
TRACE_MAX_STEPS = 5
STEADY_STATE_FIRST_STEP = 5
GLM_4_5_AIR = "/home/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
WORKLOADS = {
    "qwen3-30b": [
        "--model.name",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "--model.debug.force-balanced-routing",
        "--data.type",
        "fake",
    ],
    "glm-4.5-air-16l": [
        "--model.name",
        GLM_4_5_AIR,
        "--model.debug.num-layers",
        "16",
        "--renderer.name",
        "glm-4.5",
        "--data.type",
        "sft",
        "--data.name",
        "PrimeIntellect/INTELLECT-3-SFT-10K",
        "--data.splits",
        '["math"]',
        "--data.seed",
        "0",
        "--optim.lr",
        "1e-5",
    ],
}


def build_command(workload: str, output_dir: Path, trace_dir: Path | None) -> list[str]:
    max_steps = TRACE_MAX_STEPS if trace_dir is not None else MAX_STEPS
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        f"--nproc-per-node={NUM_GPUS}",
        "src/prime_rl/trainer/sft/train.py",
        *WORKLOADS[workload],
        "--model.seq-len",
        str(SEQ_LEN),
        "--model.attn",
        "flash_attention_3",
        "--model.ac",
        "--model.compile",
        "--model.moe-router-dtype",
        "float32",
        "--model.ep",
        str(EP),
        "--model.optim-cpu-offload",
        "false",
        "--model.fsdp-cpu-offload",
        "false",
        "--model.ac-offloading",
        "None",
        "--data.batch-size",
        str(BATCH_SIZE),
        "--data.micro-batch-size",
        "1",
        "--data.seq-len",
        str(SEQ_LEN),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
        "--run.name",
        "bench",
        "--monitors.file.path",
        "metrics.jsonl",
    ]
    if trace_dir is not None:
        cmd += ["--trace-path", str(trace_dir)]
    return cmd


def fresh_cache_env(run_name: str) -> dict[str, str]:
    cache_root = Path("/tmp") / os.environ["USER"] / "fp32-router-caches" / run_name
    shutil.rmtree(cache_root, ignore_errors=True)
    env = {key: value for key, value in os.environ.items() if key != "HF_HOME"}
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TILELANG_CACHE_DIR"):
        env[name] = str(cache_root / name.removesuffix("_CACHE_DIR").lower())
    return env


def summarize(metrics_path: Path) -> dict:
    per_step: dict[int, dict] = {}
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("step") is not None:
            per_step.setdefault(row["step"], {}).update(row)
    steady_steps = [step for step in sorted(per_step) if step >= STEADY_STATE_FIRST_STEP]
    summary = {"steps": len(steady_steps)}
    for key in ("time/step", "perf/mfu", "perf/throughput_per_gpu", "perf/peak_memory"):
        values = [per_step[step][key] for step in steady_steps]
        summary[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", help="Arm label recorded in the results, e.g. A0")
    parser.add_argument("workload", choices=sorted(WORKLOADS))
    parser.add_argument("out_dir", type=Path, help="Root directory for run outputs")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index, for same-code noise floors")
    parser.add_argument(
        "--trace", action="store_true", help=f"Capture torch.profiler traces over {TRACE_MAX_STEPS} steps"
    )
    args = parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    run_name = f"{args.workload}-{args.arm}-r{args.repeat}" + ("-trace" if args.trace else "")
    output_dir = args.out_dir / run_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    trace_dir = output_dir / "traces" if args.trace else None
    cmd = build_command(args.workload, output_dir, trace_dir)
    print(f"[{run_name}] {' '.join(cmd)}", flush=True)
    log_env = {"TORCH_LOGS": "graph_breaks,recompiles"} if args.trace else {}
    with open(output_dir / "stdout.log", "w") as log:
        returncode = subprocess.run(cmd, env=fresh_cache_env(run_name) | log_env, stdout=log, stderr=log).returncode
    result = {"run": run_name, "arm": args.arm, "commit": commit.stdout.strip(), "returncode": returncode}
    if returncode == 0 and not args.trace:
        result |= summarize(output_dir / "bench" / "monitors" / "file" / "metrics.jsonl")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
