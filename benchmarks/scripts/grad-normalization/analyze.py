"""Summarize gradient-normalization A/B runs: per-step parity between two runs, and the floored-encoder e_min sweep.

Each RUN is a run directory passed to launch.sh as OUT/NAME; probe records are read from RUN.gn/probes.jsonl and
trainer/orchestrator metrics from RUN/monitors/file/metrics.jsonl. It needs only matplotlib, e.g.
`uv run --no-project --with matplotlib python analyze.py ...`.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PRODUCTION_EMIN = -22
E4M3_SUBNORMAL_BINADES = 10
LOSS_KEY = "loss/mean"
GRAD_NORM_KEY = "optim/grad_norm"
REWARD_KEY = "train/agg/all/agent/reward/mean"
STEP_TIME_KEY = "time/step"
COLORS = {"loss": "#2a78d6", "grads": "#eb6834"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_metrics(run: Path) -> dict[int, dict]:
    by_step: dict[int, dict] = defaultdict(dict)
    for record in read_jsonl(run / "monitors" / "file" / "metrics.jsonl"):
        if record.get("step") is not None:
            producer = record.get("producer", "")
            for key, value in record.items():
                by_step[record["step"]][f"{producer}:{key}"] = value
                by_step[record["step"]].setdefault(key, value)
    return by_step


def read_probes(run: Path) -> list[dict]:
    return read_jsonl(Path(f"{run}.gn") / "probes.jsonl")


def placement(records: list[dict]) -> str:
    for record in records:
        norm = record.get("normalization")
        if norm and norm.get("placement"):
            return norm["placement"]
    return "loss"


def fmt(value, spec=".4g") -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def rel_diff(a, b):
    if a is None or b is None or a == 0:
        return None
    return abs(b - a) / abs(a)


def layer_mean(records: list[dict], site: str, field: str) -> dict[int, float]:
    values = defaultdict(list)
    for record in records:
        for act in record["acts"]:
            if act["site"] == site:
                values[act["layer"]].append(act[field])
    return {layer: statistics.fmean(v) for layer, v in sorted(values.items())}


def param_ratio(records: list[dict], numerator: str, denominator: str, field: str) -> dict[int, float]:
    ratios = defaultdict(list)
    for record in records:
        by_key = {(p["layer"], p["group"]): p for p in record["params"]}
        for (layer, group), stats in by_key.items():
            other = by_key.get((layer, denominator))
            if group == numerator and other and stats.get(field) and other.get(field):
                ratios[layer].append(stats[field] / other[field])
    return {layer: statistics.fmean(v) for layer, v in sorted(ratios.items())}


def param_field(records: list[dict], group: str, field: str) -> dict[int, float]:
    values = defaultdict(list)
    for record in records:
        for stats in record["params"]:
            if stats["group"] == group and stats.get(field) is not None:
                values[stats["layer"]].append(stats[field])
    return {layer: statistics.fmean(v) for layer, v in sorted(values.items())}


def check_once_per_micro_batch(records: list[dict]) -> list[str]:
    problems = []
    for record in records:
        for act in record["acts"]:
            if act["calls"] != record["micro_batches"]:
                problems.append(
                    f"step {record['step']} layer {act['layer']} {act['site']}: "
                    f"{act['calls']} calls for {record['micro_batches']} micro batches"
                )
    return problems


def parity(args):
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    metrics_a, metrics_b = read_metrics(run_a), read_metrics(run_b)
    probes_a, probes_b = read_probes(run_a), read_probes(run_b)
    lines = [f"A: `{run_a.name}` ({placement(probes_a)}), B: `{run_b.name}` ({placement(probes_b)})", ""]
    lines += [
        "| step | loss A | loss B | rel diff | reward A | reward B | grad norm A | grad norm B | rel diff |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in sorted(set(metrics_a) | set(metrics_b)):
        a, b = metrics_a.get(step, {}), metrics_b.get(step, {})
        loss_a, loss_b = a.get(f"trainer:{LOSS_KEY}"), b.get(f"trainer:{LOSS_KEY}")
        norm_a, norm_b = a.get(f"trainer:{GRAD_NORM_KEY}"), b.get(f"trainer:{GRAD_NORM_KEY}")
        if loss_a is None and loss_b is None:
            continue
        lines.append(
            f"| {step} | {fmt(loss_a)} | {fmt(loss_b)} | {fmt(rel_diff(loss_a, loss_b), '.2e')} "
            f"| {fmt(a.get(f'orchestrator:{REWARD_KEY}'))} | {fmt(b.get(f'orchestrator:{REWARD_KEY}'))} "
            f"| {fmt(norm_a)} | {fmt(norm_b)} | {fmt(rel_diff(norm_a, norm_b), '.2e')} |"
        )
    times = {
        name: [m[f"trainer:{STEP_TIME_KEY}"] for s, m in metrics.items() if s > 1 and f"trainer:{STEP_TIME_KEY}" in m]
        for name, metrics in (("A", metrics_a), ("B", metrics_b))
    }
    lines.append("")
    for name, values in times.items():
        if values:
            lines.append(
                f"Trainer step time {name}, steps after the first (lower is better): median {statistics.median(values):.2f} s, "
                f"min {min(values):.2f} s, max {max(values):.2f} s"
            )

    if probes_a and probes_b:
        rms_a, rms_b = layer_mean(probes_a, "dy", "rms"), layer_mean(probes_b, "dy", "rms")
        down_a = param_ratio(probes_a, "routed_down", "shared", "grad_rms")
        down_b = param_ratio(probes_b, "routed_down", "shared", "grad_rms")
        gate_up_a = param_ratio(probes_a, "routed_gate_up", "shared", "grad_rms")
        gate_up_b = param_ratio(probes_b, "routed_gate_up", "shared", "grad_rms")
        zeros_a, zeros_b = param_field(probes_a, "routed_down", "grad_zero_frac"), param_field(probes_b, "routed_down", "grad_zero_frac")
        lines += [
            "",
            "Per MoE layer, mean over steps. down/shared and gate_up/shared are the RMS of the routed expert "
            "gradient over the shared expert's gradient RMS, after the post-backward scale. zero is the fraction of "
            "exactly-zero routed down-projection gradient entries.",
            "",
            "| layer | log2 rms dy A | log2 rms dy B | down/shared A | down/shared B | gate_up/shared A "
            "| gate_up/shared B | zero A | zero B |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for layer in sorted(rms_a):
            lines.append(
                f"| {layer} | {fmt(math.log2(rms_a[layer]), '.2f')} "
                f"| {fmt(math.log2(rms_b[layer]) if rms_b.get(layer) else None, '.2f')} "
                f"| {fmt(down_a.get(layer))} | {fmt(down_b.get(layer))} | {fmt(gate_up_a.get(layer))} "
                f"| {fmt(gate_up_b.get(layer))} | {fmt(zeros_a.get(layer), '.3g')} | {fmt(zeros_b.get(layer), '.3g')} |"
            )
        for name, probes in (("A", probes_a), ("B", probes_b)):
            seeds = [r["seed"]["ratio_times_n_quantiles"]["0.5"] for r in probes if r.get("seed", {}).get("ratio_times_n_quantiles")]
            tokens = [r["global_tokens"] for r in probes]
            problems = check_once_per_micro_batch(probes)
            lines.append("")
            lines.append(
                f"{name}: global loss tokens per step {fmt(statistics.fmean(tokens), '.0f')}; median "
                f"|dL/dlogp| / (|w A| / N) per step {', '.join(fmt(s, '.3g') for s in seeds)}; "
                f"probe calls per micro batch {'all 1' if not problems else 'MISMATCH: ' + '; '.join(problems[:3])}"
            )
    write(args.out, "\n".join(lines))


def sweep_rows(runs: list[Path], site: str) -> list[dict]:
    rows = []
    for run in runs:
        records = read_probes(run)
        if not records:
            continue
        first = records[0]
        tokens = statistics.fmean(r["global_tokens"] for r in records)
        rms = layer_mean(records, site, "rms")
        p50 = layer_mean(records, site, "p50")
        for e_min in first["emin_scan"]:
            zeroed = layer_mean(
                [{**r, "acts": [{**a, "z": a["zeroed_frac"][str(e_min)]} for a in r["acts"]]} for r in records], site, "z"
            )
            floored = layer_mean(
                [{**r, "acts": [{**a, "f": a["floor_block_frac"][str(e_min)]} for a in r["acts"]]} for r in records],
                site,
                "f",
            )
            rows.append(
                {
                    "run": run.name,
                    "placement": placement(records),
                    "e_min": e_min,
                    "applied": first["encoder"] and e_min == first["e_min"] and site in first["encoder_sites"],
                    "tokens": tokens,
                    "margin": {layer: math.log2(v) - (e_min - E4M3_SUBNORMAL_BINADES) for layer, v in rms.items()},
                    "p50_margin": {
                        layer: math.log2(v) - (e_min - E4M3_SUBNORMAL_BINADES) for layer, v in p50.items() if v > 0
                    },
                    "zeroed": zeroed,
                    "floored": floored,
                    "log2_rms": {layer: math.log2(v) for layer, v in rms.items()},
                }
            )
    return rows


def sweep(args):
    runs = [Path(run) for run in args.runs]
    rows = sweep_rows(runs, args.site)
    lines = [
        f"Site `{args.site}`. Margin = log2(rms) - (e_min - {E4M3_SUBNORMAL_BINADES}); the p50 margin uses the median "
        "|g| instead of the rms. Encoder `applied` rows train "
        "through the emulated encoder; the others only measure what it would zero. With power-of-two scales, a run "
        f"with N tokens per step at e_min matches a `loss`-placement run with N * 2^(e_min - ({PRODUCTION_EMIN})) tokens "
        f"at e_min = {PRODUCTION_EMIN} (the equivalent N column); `grads` placement does not depend on N.",
        "",
        "| run | placement | e_min | applied | N | equivalent N at -22 | min margin | min p50 margin "
        "| max zeroed | median zeroed | median floored blocks |",
        "| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        margins, zeroed = list(row["margin"].values()), list(row["zeroed"].values())
        equivalent = row["tokens"] * 2.0 ** (row["e_min"] - PRODUCTION_EMIN) if row["placement"] == "loss" else None
        lines.append(
            f"| {row['run']} | {row['placement']} | {row['e_min']} | {'yes' if row['applied'] else ''} "
            f"| {row['tokens']:.0f} | {fmt(equivalent, '.2e')} | {min(margins):.2f} "
            f"| {fmt(min(row['p50_margin'].values(), default=None), '.2f')} "
            f"| {max(zeroed):.3g} | {statistics.median(zeroed):.3g} "
            f"| {statistics.median(row['floored'].values()):.3g} |"
        )
    seen = {}
    for row in rows:
        seen.setdefault((row["run"], row["placement"]), row["log2_rms"])
    lines += ["", f"log2 rms of `{args.site}` per MoE layer (mean over steps):", ""]
    header = list(seen)
    lines.append("| layer | " + " | ".join(f"{run} ({p})" for run, p in header) + " |")
    lines.append("| ---: |" + " ---: |" * len(header))
    for layer in sorted(next(iter(seen.values()), {})):
        lines.append(f"| {layer} | " + " | ".join(fmt(seen[key].get(layer), ".2f") for key in header) + " |")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / f"sweep_{args.site}.md", "\n".join(lines))
    plot_sweep(rows, args.site, out_dir / f"sweep_{args.site}.png")


def plot_sweep(rows: list[dict], site: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (zero_ax, rms_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    for place, color in COLORS.items():
        selected = sorted((r for r in rows if r["placement"] == place), key=lambda r: r["e_min"])
        if not selected:
            continue
        e_mins = [r["e_min"] for r in selected]
        zero_ax.plot(e_mins, [max(r["zeroed"].values()) for r in selected], color=color, lw=2, marker="o", ms=8,
                     label=f"{place}: max over layers")
        zero_ax.plot(e_mins, [statistics.median(r["zeroed"].values()) for r in selected], color=color, lw=2,
                     ls="--", marker="o", ms=8, label=f"{place}: median over layers")
        log2_rms = selected[0]["log2_rms"]
        rms_ax.plot(list(log2_rms), list(log2_rms.values()), color=color, lw=2, marker="o", ms=8, label=place)
    zero_ax.set_xlabel("e_min (block-scale exponent floor)")
    zero_ax.set_ylabel(f"fraction of nonzero {site} entries zeroed")
    zero_ax.set_title(f"{site} zeroed by the floored E4M3 encoder")
    rms_ax.axhline(PRODUCTION_EMIN - E4M3_SUBNORMAL_BINADES, color="#52514e", lw=1, ls=":",
                   label=f"zeroing threshold at e_min = {PRODUCTION_EMIN}")
    rms_ax.set_xlabel("layer")
    rms_ax.set_ylabel(f"log2 rms {site}")
    rms_ax.set_title(f"{site} magnitude per MoE layer")
    for ax in (zero_ax, rms_ax):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def write(path, text: str):
    if path is None:
        print(text)
        return
    Path(path).write_text(text + "\n")
    print(text)
    print(f"\nwrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    parity_parser = sub.add_parser("parity", help="per-step loss, reward and grad norm of two runs, plus probe stats")
    parity_parser.add_argument("run_a")
    parity_parser.add_argument("run_b")
    parity_parser.add_argument("--out", help="markdown output file (default: stdout only)")
    parity_parser.set_defaults(func=parity)
    sweep_parser = sub.add_parser("sweep", help="margin and zeroed fraction per placement and e_min")
    sweep_parser.add_argument("runs", nargs="+")
    sweep_parser.add_argument("--site", default="dy", choices=["dy", "dy_rows", "dz", "sdy", "sdz", "ao", "aqkv"])
    sweep_parser.add_argument("--out-dir", required=True)
    sweep_parser.set_defaults(func=sweep)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
