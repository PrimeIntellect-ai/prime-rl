"""Per-arm steady-state summary and per-step table from each run's monitors/file/metrics.jsonl."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path.home() / "tmp/sft_pad_to_cp/runs"


def load_metrics(run: str) -> dict[int, dict]:
    by_step = defaultdict(dict)
    for line in (RUNS_DIR / run / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_step[record["step"]].update(record)
    return dict(sorted(by_step.items()))


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--warmup", type=int, default=4, help="Leading steps excluded from the summary.")
    parser.add_argument("--per-step", action="store_true")
    args = parser.parse_args()

    print(
        f"{'run':<36} {'steps':>5} {'step s, median [min, max]':>26} {'fwd/bwd s':>22} "
        f"{'real tok/s/GPU':>14} {'samples/s':>9} {'tokens M':>8} {'peak GiB':>8} {'tunes':>5} {'graphs':>6} {'alloc retries':>13}"
    )
    for run in args.runs:
        metrics = load_metrics(run)
        steady = [m for step, m in metrics.items() if step > args.warmup]
        samples = [m["progress/num_samples"] for m in metrics.values()]
        steady_time = sum(m["time/step"] for m in steady)
        samples_per_s = (samples[-1] - samples[args.warmup - 1]) / steady_time if len(samples) > args.warmup else 0
        tokens = [m["progress/num_tokens"] for m in metrics.values()]
        steady_tokens_m = (tokens[-1] - tokens[args.warmup - 1]) / 1e6 if len(tokens) > args.warmup else 0
        last = list(metrics.values())[-1]
        print(
            f"{run:<36} {len(steady):>5} {spread([m['time/step'] for m in steady]):>26} "
            f"{spread([m['time/forward_backward'] for m in steady]):>22} "
            f"{statistics.median(m['perf/throughput_per_gpu'] for m in steady):>14.0f} {samples_per_s:>9.2f} {steady_tokens_m:>8.2f} "
            f"{max(m['perf/peak_memory'] for m in steady):>8.1f} {last.get('diag/indexer_tuning_keys', '-'):>5} "
            f"{last.get('diag/dynamo_unique_graphs', '-'):>6} {last.get('diag/num_alloc_retries', '-'):>13}"
        )
        if args.per_step:
            for step, m in metrics.items():
                print(
                    f"  step {step:>2} row {m.get('diag/row_len', '-'):>7} step {m['time/step']:6.2f}s "
                    f"fwd/bwd {m['time/forward_backward']:6.2f}s loss {m['loss/mean']:.5f} "
                    f"gnorm {m.get('optim/grad_norm', float('nan')):.4f} samples {m['progress/num_samples']:>4} "
                    f"tunes {m.get('diag/indexer_tuning_keys', '-')} graphs {m.get('diag/dynamo_unique_graphs', '-')} "
                    f"dev_alloc {m.get('diag/num_device_alloc', '-')} retries {m.get('diag/num_alloc_retries', '-')}"
                )


if __name__ == "__main__":
    main()
