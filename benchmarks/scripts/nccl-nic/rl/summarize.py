"""Summarize the RL A/B runs: NCCL versions per component, step and broadcast times, inference throughput, reward and loss trend, and per-component exit times."""

import argparse
import json
import re
import statistics
from pathlib import Path

STEP_LINE = re.compile(r"(?<!Micro )Step (\d+) \|")


def median_range(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.median(values):.2f} ({min(values):.2f}, {max(values):.2f})"


def read_metrics(run_dir: Path) -> tuple[dict[int, dict], list[dict]]:
    trainer: dict[int, dict] = {}
    orchestrator: list[dict] = []
    path = run_dir / "monitors" / "file" / "metrics.jsonl"
    if not path.exists():
        return trainer, orchestrator
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("producer") == "trainer" and "step" in row:
            trainer.setdefault(row["step"], {}).update(row)
        elif row.get("producer") == "orchestrator":
            orchestrator.append(row)
    return trainer, orchestrator


def nccl_versions(bench: Path) -> dict[str, list[str]]:
    versions: dict[str, set[str]] = {}
    logs = {"trainer": sorted((bench / "ranks").glob("rank_*.log"))}
    logs.update({f.stem: [f] for f in sorted(bench.glob("vllm_*.log"))})
    for name, files in logs.items():
        found: set[str] = set()
        for f in files:
            text = f.read_text(errors="replace")
            found.update(re.findall(r"NCCL version (\S+)", text))
            found.update(f"vllm-pynccl {v}" for v in re.findall(r"vLLM is using nccl==(\S+)", text))
        versions[name] = sorted(found)
    return versions


def step_epoch(log: Path, step: int) -> float | None:
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        match = STEP_LINE.search(line)
        if match and int(match.group(1)) == step:
            return float(line.split(" ", 1)[0])
    return None


def exit_record(path: Path) -> dict[str, str]:
    return dict(re.findall(r"(\w+) (\S+)", path.read_text())) if path.exists() else {}


def summarize(run_dir: Path, warmup: int) -> None:
    bench = run_dir / "bench"
    print(f"## {run_dir.name}")
    print((bench / "run_info.txt").read_text().strip())
    for node in sorted(bench.glob("node_*.txt")):
        print(node.read_text().strip())
    print("NCCL banners:", json.dumps(nccl_versions(bench)))

    trainer, orchestrator = read_metrics(run_dir)
    steps = sorted(trainer)
    steady = [trainer[s] for s in steps if s > warmup]
    print(f"trainer steps logged: {len(steps)} (last {steps[-1] if steps else None})")
    print("trainer time/step median (min, max) after warmup:", median_range([r["time/step"] for r in steady]))
    print("trainer time/broadcast_weights median (min, max), all steps:", median_range([trainer[s]["time/broadcast_weights"] for s in steps if trainer[s].get("time/broadcast_weights")]))
    print("trainer time/forward_backward median after warmup:", median_range([r["time/forward_backward"] for r in steady]))
    print("trainer time/wait_for_batch median after warmup:", median_range([r["time/wait_for_batch"] for r in steady]))
    transfer = [r["watcher/transfer/max"] for r in orchestrator if r.get("watcher/transfer/max")]
    print("orchestrator watcher/transfer/max median (min, max):", median_range(transfer))
    rate = [r["inference/agg/generation_tokens_total:rate/sum"] for r in orchestrator if r.get("inference/agg/generation_tokens_total:rate/sum")]
    print("inference generation tokens/s (sum over engines, nonzero samples) median (min, max):", median_range(rate))
    print("per step: step, loss, mismatch_kl, entropy, grad_norm, broadcast s, step s")
    for s in steps:
        r = trainer[s]
        print(
            f"  {s:>2} {r.get('loss/mean', float('nan')):+.4f} {r.get('mismatch_kl/all/mean', float('nan')):.5f} "
            f"{r.get('entropy/all/mean', float('nan')):.4f} {r.get('optim/grad_norm', float('nan')):.4f} "
            f"{r.get('time/broadcast_weights', float('nan')):.2f} {r.get('time/step', float('nan')):.2f}"
        )
    rewards = []
    for r in orchestrator:
        for key, value in r.items():
            if key.startswith("train/agg/effective/") and key.endswith("/reward/mean") and "step" in r:
                rewards.append((r["step"], value))
    print("orchestrator reward/mean by step:", " ".join(f"{s}:{v:.3f}" for s, v in sorted(dict(rewards).items())))

    last_step = steps[-1] if steps else None
    anchor = step_epoch(bench / "ranks" / "rank_0.log", last_step) if last_step is not None else None
    print(f"last trainer step epoch: {anchor}")
    rank_exits = []
    for exit_file in sorted((bench / "ranks").glob("exit_rank_*.txt"), key=lambda p: int(p.stem.split("_")[-1])):
        rec = exit_record(exit_file)
        if "exit" in rec and anchor is not None:
            rank_exits.append(f"r{exit_file.stem.split('_')[-1]}:{float(rec['exit']) - anchor:.1f}s/rc{rec['rc']}")
        else:
            rank_exits.append(f"r{exit_file.stem.split('_')[-1]}:NO_EXIT")
    print("trainer rank exit after last step:", " ".join(rank_exits))
    for name in ("trainer_torchrun", "orchestrator"):
        rec = exit_record(bench / "exits" / f"{name}.txt")
        after = f"{float(rec['exit']) - anchor:.1f}s" if "exit" in rec and anchor is not None else "n/a"
        print(f"{name}: exit after last step {after}, rc {rec.get('rc')}")
    teardown = bench / "infer_teardown.txt"
    if teardown.exists():
        print("inference teardown (SIGTERM after trainer node finished):")
        for line in teardown.read_text().splitlines():
            print("  " + line)
    for dump in sorted(bench.glob("hang_dump_*.txt")):
        print(f"hang dump present: {dump}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    for run in args.runs:
        summarize(args.runs_root / run, args.warmup)


if __name__ == "__main__":
    main()
