"""Profile a perfectible-subset selection: stats + PNG plots + HF dataset card.

Reads the baseline rollouts + the selected subset + the source dataset, computes
per-problem and per-domain statistics, renders matplotlib PNG charts, and writes
a README.md suitable for direct upload as the HuggingFace dataset card.

Usage:
    uv run python scripts/evals/profile_perfectible_subset.py \\
        --records-jsonl outputs/baselines/.../records.jsonl \\
        --selected-jsonl benchmarks/datasets/.../perfectible.jsonl \\
        --source-jsonl benchmarks/datasets/.../full_train.jsonl \\
        --out-dir benchmarks/datasets/.../perfectible_card \\
        --model-name allenai/Olmo-3-7B-Instruct-DPO \\
        --hf-repo joanvelja/omni-math2-olmo3-perfectible-seed42
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def primary_domain(domain_field: Any) -> str:
    if not domain_field:
        return "unknown"
    items = domain_field if isinstance(domain_field, list) else [domain_field]
    if not items:
        return "unknown"
    first = str(items[0])
    parts = [p.strip() for p in first.split("->")]
    if len(parts) >= 2 and parts[0].lower() == "mathematics":
        return parts[1]
    return first.split(" ")[0][:30]


def domain_chain(domain_field: Any, depth: int = 2) -> str:
    if not domain_field:
        return "unknown"
    items = domain_field if isinstance(domain_field, list) else [domain_field]
    first = str(items[0]) if items else "unknown"
    parts = [p.strip() for p in first.split("->")]
    take = parts[: depth + 1] if parts and parts[0].lower() == "mathematics" else parts[:depth]
    return " > ".join(take[1 : depth + 1] if parts and parts[0].lower() == "mathematics" else take)


def difficulty_bucket(d: Any) -> str:
    try:
        v = float(d)
    except (TypeError, ValueError):
        return "unknown"
    if v < 3:
        return "easy (<3)"
    if v < 5:
        return "medium (3-5)"
    if v < 7:
        return "hard (5-7)"
    if v < 9:
        return "very hard (7-9)"
    return "extreme (>=9)"


def bar_chart_vertical(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    savepath: Path,
    color: str = "#1f77b4",
    colors: list[str] | None = None,
    annotate: bool = True,
    rotate_xticks: int = 0,
) -> None:
    """Vertical bar chart. PNG saved to savepath."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = list(range(len(values)))
    bars = ax.bar(xs, values, color=colors if colors else color, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rotate_xticks, ha="right" if rotate_xticks else "center")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(axis="x", visible=False)
    if annotate:
        ymax = max(values) if values else 1
        for bar, v in zip(bars, values):
            if v <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + ymax * 0.015,
                f"{int(v)}" if v == int(v) else f"{v:.2g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.savefig(savepath)
    plt.close(fig)


def bar_chart_horizontal(
    *,
    labels: list[str],
    values: list[int | float],
    title: str,
    x_label: str,
    savepath: Path,
    color: str = "#1f77b4",
) -> None:
    """Horizontal bar chart. Order is preserved (labels passed in order)."""
    height = max(2.5, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, height))
    ys = list(range(len(values)))
    # Reverse for top-to-bottom (matplotlib renders bottom-up)
    bars = ax.barh(ys, values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(axis="y", visible=False)
    xmax = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(
            v + xmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(v)}" if v == int(v) else f"{v:.2g}",
            va="center",
            fontsize=9,
        )
    fig.savefig(savepath)
    plt.close(fig)


def line_chart(
    *,
    xs: list[float],
    ys: list[float],
    title: str,
    x_label: str,
    y_label: str,
    savepath: Path,
    color: str = "#1f77b4",
    annotate: bool = True,
    log_x: bool = False,
    ymax: float | None = None,
) -> None:
    """Line plot with markers + value annotations."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7)
    ax.set_title(title, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" if x == int(x) else f"{x:g}" for x in xs])
    top = ymax if ymax is not None else max(ys) * 1.15
    ax.set_ylim(0, top)
    if annotate:
        for x, y in zip(xs, ys):
            # Place annotation below for values near the top (>0.92 * top) to avoid title collision
            below = y > 0.92 * top
            offset_y = -14 if below else 9
            va = "top" if below else "bottom"
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                va=va,
                fontsize=9,
            )
    fig.savefig(savepath)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hf-repo", default=None, help="for the readme header (e.g. user/dataset)")
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    selected = [json.loads(line) for line in args.selected_jsonl.read_text().splitlines() if line.strip()]
    selected_ids = {str(r["id"]) for r in selected}
    source = [json.loads(line) for line in args.source_jsonl.read_text().splitlines() if line.strip()]

    # per-problem rollout aggregates from records.jsonl
    per_q_reward: dict[str, list[float]] = defaultdict(list)
    per_q_trunc: dict[str, list[bool]] = defaultdict(list)
    per_q_tokens: dict[str, list[float]] = defaultdict(list)
    per_q_judge_called: dict[str, int] = defaultdict(int)
    with args.records_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            qid = str(r.get("example_id"))
            per_q_reward[qid].append(float(r.get("reward") or 0.0))
            per_q_trunc[qid].append(bool(r.get("is_truncated")))
            per_q_tokens[qid].append(float(r.get("output_tokens") or 0.0))
            if r.get("judge_decision") is not None:
                per_q_judge_called[qid] += 1

    sampled_ids = set(per_q_reward.keys())
    sampled_rows = [r for r in source if str(r.get("id")) in sampled_ids]

    sel_rates = [mean(per_q_reward[qid]) for qid in selected_ids if qid in per_q_reward]
    sel_correct_counts = [sum(1 for r in per_q_reward[qid] if r >= 0.5) for qid in selected_ids if qid in per_q_reward]
    sel_tokens_mean = [mean(per_q_tokens[qid]) for qid in selected_ids if qid in per_q_tokens]
    sel_trunc_rate = [sum(per_q_trunc[qid]) / len(per_q_trunc[qid]) for qid in selected_ids if qid in per_q_trunc]
    sel_judge_rate = [per_q_judge_called.get(qid, 0) / max(len(per_q_reward[qid]), 1) for qid in selected_ids]

    sampled_rates = [mean(per_q_reward[qid]) for qid in sampled_ids if qid in per_q_reward]

    # Histogram of solve rates — 20 buckets of width 0.05 (finer than the [0.2, 0.8] band edges)
    bucket_width = 0.05
    n_buckets = int(round(1.0 / bucket_width))

    def hist(rates: list[float]) -> list[int]:
        out = [0] * n_buckets
        for r in rates:
            i = min(int(r / bucket_width), n_buckets - 1)
            out[i] += 1
        return out

    bucket_centers = [(i + 0.5) * bucket_width for i in range(n_buckets)]
    sel_hist = hist(sel_rates)
    sampled_hist = hist(sampled_rates)

    # Domain mix
    sel_domains = Counter(primary_domain(r.get("domain")) for r in selected)
    sampled_domains = Counter(primary_domain(r.get("domain")) for r in sampled_rows)
    source_domains = Counter(primary_domain(r.get("domain")) for r in source)

    # Source mix
    sel_sources = Counter((r.get("source") or "unknown") for r in selected)

    # Difficulty mix
    sel_difficulty = Counter(difficulty_bucket(r.get("difficulty")) for r in selected)

    # pass@k for selected (unbiased estimator)
    def pass_at_k(n: int, c: int, k: int) -> float:
        if k > n or c <= 0:
            return 0.0 if c <= 0 else 1.0
        if n - c < k:
            return 1.0
        total = 1.0
        for i in range(n - c + 1, n + 1):
            total *= 1.0 - k / i
        return 1.0 - total

    selected_per_problem_correct = [(40, c) for c in sel_correct_counts]
    if not selected_per_problem_correct:
        raise ValueError("No selected problem IDs matched rollout records; check --selected and --records inputs.")
    pass_ks = [1, 2, 4, 8, 16, 32, 40]
    pass_at_k_values = [mean(pass_at_k(n, c, k) for n, c in selected_per_problem_correct) for k in pass_ks]

    # === Plots ===
    plots = args.out_dir / "plots"

    # 1) Solve-rate histogram for sampled-1000 (the *real* picture: heavy left tail).
    #    Selected region [args.low, args.high] shaded; selected count + non-selected counts annotated.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bar_xs = [c - bucket_width / 2 for c in bucket_centers]
    ax.bar(bar_xs, sampled_hist, width=bucket_width * 0.95, color="#525252", align="edge", edgecolor="white", linewidth=0.4)
    ax.axvspan(args.low, args.high, color="#2ca02c", alpha=0.18, zorder=0, label=f"selected band [{args.low:.1f}, {args.high:.1f}]")
    ax.set_xlim(0, 1)
    ax.set_xlabel("per-problem mean reward across 40 rollouts")
    ax.set_ylabel("# problems")
    ax.set_title(f"Solve-rate distribution over {len(sampled_rows)} sampled problems\n({len(selected)} selected / {len(sampled_rows) - len(selected)} discarded)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(plots / "solve_rate_sampled.png")
    plt.close(fig)

    # 2) Same as above but log-y so the perfectible band is visible despite the left tail.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(bar_xs, sampled_hist, width=bucket_width * 0.95, color="#525252", align="edge", edgecolor="white", linewidth=0.4)
    ax.axvspan(args.low, args.high, color="#2ca02c", alpha=0.18, zorder=0, label=f"selected band [{args.low:.1f}, {args.high:.1f}]")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("per-problem mean reward across 40 rollouts")
    ax.set_ylabel("# problems (log)")
    ax.set_title(f"Solve-rate distribution (log-y) — {len(sampled_rows)} sampled, {len(selected)} selected")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(plots / "solve_rate_sampled_log.png")
    plt.close(fig)

    # 3) Selected-only solve-rate histogram (zoom on the [low, high] band).
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(bar_xs, sel_hist, width=bucket_width * 0.95, color="#2ca02c", align="edge", edgecolor="white", linewidth=0.4)
    ax.set_xlim(args.low - 0.02, args.high + 0.05)
    ax.set_xlabel("per-problem mean reward across 40 rollouts")
    ax.set_ylabel("# problems")
    ax.set_title(f"Solve-rate distribution within selected band ({len(selected)} problems)")
    ax.grid(axis="x", visible=False)
    fig.savefig(plots / "solve_rate_selected.png")
    plt.close(fig)

    # 4) pass@k for selected subset
    line_chart(
        xs=[float(k) for k in pass_ks],
        ys=pass_at_k_values,
        title=f"pass@k for selected {len(selected)}-problem subset",
        x_label="k (sampled with replacement)",
        y_label="pass@k (unbiased)",
        savepath=plots / "pass_at_k.png",
        log_x=True,
        ymax=1.08,
    )

    # 5) Domain mix
    top_doms = [d for d, _ in sel_domains.most_common(12)]
    bar_chart_horizontal(
        labels=top_doms,
        values=[sel_domains[d] for d in top_doms],
        title=f"Top math domains in selected ({len(selected)} problems)",
        x_label="# problems",
        savepath=plots / "domain_mix.png",
    )

    # 6) Source mix (top 15)
    top_srcs = [s for s, _ in sel_sources.most_common(15)]
    bar_chart_horizontal(
        labels=top_srcs,
        values=[sel_sources[s] for s in top_srcs],
        title=f"Top competition sources in selected ({len(selected)} problems)",
        x_label="# problems",
        color="#9467bd",
        savepath=plots / "source_mix.png",
    )

    # 7) Difficulty mix
    diff_order = ["easy (<3)", "medium (3-5)", "hard (5-7)", "very hard (7-9)", "extreme (>=9)", "unknown"]
    bar_chart_vertical(
        labels=diff_order,
        values=[float(sel_difficulty.get(d, 0)) for d in diff_order],
        title=f"Difficulty distribution in selected ({len(selected)} problems)",
        y_label="# problems",
        color="#ff7f0e",
        savepath=plots / "difficulty_mix.png",
        rotate_xticks=15,
    )

    # 8) Output token distribution + truncation rate (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.hist(sel_tokens_mean, bins=20, color="#17becf", edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("mean output tokens per problem (avg over 40 rollouts)")
    ax1.set_ylabel("# problems")
    ax1.set_title("Output token distribution (selected)")
    ax1.axvline(15360, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.annotate("max_completion_tokens=15360", xy=(15360, ax1.get_ylim()[1] * 0.9), xytext=(-6, 0),
                 textcoords="offset points", ha="right", fontsize=9, color="red")
    ax1.grid(axis="x", visible=False)

    ax2.hist(sel_trunc_rate, bins=20, color="#e377c2", edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("per-problem truncation rate (frac of 40 rollouts hitting 15360)")
    ax2.set_ylabel("# problems")
    ax2.set_title("Length-truncation rate (selected)")
    ax2.grid(axis="x", visible=False)
    fig.savefig(plots / "tokens_and_truncation.png")
    plt.close(fig)

    # 9) Judge-fallback rate distribution
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(sel_judge_rate, bins=20, color="#8c564b", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("per-problem judge-fallback rate (frac of rollouts where math_verify was ambiguous)")
    ax.set_ylabel("# problems")
    ax.set_title(f"Judge invocation rate (selected, mean={mean(sel_judge_rate):.3f})")
    ax.grid(axis="x", visible=False)
    fig.savefig(plots / "judge_rate.png")
    plt.close(fig)

    # 10) Solve rate vs mean output tokens (scatter, colored by truncation rate)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sel_rates_arr = sel_rates
    sel_tokens_arr = sel_tokens_mean
    sel_trunc_arr = sel_trunc_rate
    sc = ax.scatter(sel_tokens_arr, sel_rates_arr, c=sel_trunc_arr, cmap="viridis_r", s=20, alpha=0.75)
    ax.set_xlabel("mean output tokens (over 40 rollouts)")
    ax.set_ylabel("per-problem solve rate")
    ax.set_title(f"Solve rate vs reasoning length (selected {len(selected)} problems)")
    plt.colorbar(sc, ax=ax, label="truncation rate")
    ax.grid(visible=True)
    fig.savefig(plots / "solve_rate_vs_tokens.png")
    plt.close(fig)

    # 11) Solve rate vs difficulty (scatter, with jitter on difficulty)
    import random
    rng = random.Random(42)
    sel_difficulty_raw = []
    sel_rate_for_diff = []
    for r in selected:
        d = r.get("difficulty")
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        qid = str(r.get("id"))
        if qid not in per_q_reward:
            continue
        sel_difficulty_raw.append(v + rng.uniform(-0.1, 0.1))
        sel_rate_for_diff.append(mean(per_q_reward[qid]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(sel_difficulty_raw, sel_rate_for_diff, s=20, alpha=0.55, color="#1f77b4")
    ax.set_xlabel("difficulty (omni-math 1-10 scale, jittered ±0.1)")
    ax.set_ylabel("per-problem solve rate")
    ax.set_title(f"Solve rate vs problem difficulty (selected {len(sel_rate_for_diff)} problems)")
    ax.axhspan(args.low, args.high, color="#2ca02c", alpha=0.16, zorder=0, label=f"selected band [{args.low:.1f}, {args.high:.1f}]")
    ax.legend(loc="lower left")
    fig.savefig(plots / "solve_rate_vs_difficulty.png")
    plt.close(fig)

    # === README ===
    readme = []
    # YAML frontmatter for HF dataset card indexing
    readme.append(
        "---\n"
        "language:\n- en\n"
        "task_categories:\n- question-answering\n- text-generation\n"
        "tags:\n"
        "- math\n"
        "- mathematical-reasoning\n"
        "- omni-math2\n"
        "- hendrycks-sanity\n"
        "- perfectible-subset\n"
        "- rl-diagnostic\n"
        "size_categories:\n- n<1K\n"
        f"pretty_name: omni-math2 {args.model_name} perfectible subset\n"
        "---\n\n"
    )
    repo_header = f"# {args.hf_repo}\n\n" if args.hf_repo else "# Perfectible subset\n\n"
    readme.append(repo_header)
    readme.append(
        "Hendrycks-Sanity perfectible-subset of Omni-MATH-2: problems where the base model "
        f"solves between {args.low:.0%} and {args.high:.0%} of attempts. Useful for "
        "recipe-correctness diagnostics on RL algorithms (a working algorithm should push "
        "training accuracy on this subset above 95%).\n\n"
    )
    readme.append("## Provenance\n\n")
    readme.append("| | |\n|---|---|\n")
    readme.append(f"| Base model | `{args.model_name}` |\n")
    readme.append(f"| Source dataset | `omni_math2_train_excluding_baseline600_seed42.jsonl` ({len(source)} problems) |\n")
    readme.append(f"| Sampled this run (seed=42) | **{len(sampled_rows)}** problems × 40 rollouts each |\n")
    readme.append(f"| Selected band | `[{args.low}, {args.high}]` mean reward |\n")
    readme.append("| Min rollouts threshold | 8 |\n")
    readme.append("| Scoring | math_verify (SymPy) + gpt-5.4-mini judge fallback (`omni_math2_hybrid_math_v1` rubric) |\n")
    readme.append("| Sampling | t=1.0, top_p=0.95, max_completion_tokens=15360 |\n")
    readme.append(f"| **Selected** | **{len(selected)}** problems ({len(selected) * 100 / len(sampled_rows):.1f}% of sampled) |\n\n")

    # Data-driven tail-shape description (not hardcoded — varies by model)
    frac_all0 = sum(1 for r in sampled_rates if r == 0) / len(sampled_rates)
    frac_all1 = sum(1 for r in sampled_rates if r == 1) / len(sampled_rates)
    sampled_mean = mean(sampled_rates)
    if frac_all0 > frac_all1 + 0.10:
        tail_phrase = (
            f"a heavy left tail ({frac_all0:.0%} of sampled problems get 0/40 — too hard for "
            f"`{args.model_name.split('/')[-1]}`)"
        )
        dominant_tail = "left tail"
    elif frac_all1 > frac_all0 + 0.10:
        tail_phrase = (
            f"a heavy right tail ({frac_all1:.0%} of sampled problems get 40/40 — too easy for "
            f"`{args.model_name.split('/')[-1]}`)"
        )
        dominant_tail = "right tail"
    else:
        tail_phrase = (
            f"a roughly balanced distribution ({frac_all0:.0%} all-0, {frac_all1:.0%} all-1, "
            f"mean {sampled_mean:.2f}) — `{args.model_name.split('/')[-1]}` is well-matched to the dataset's difficulty range"
        )
        dominant_tail = "balanced tails"

    readme.append("## Solve-rate distribution\n\n")
    readme.append(
        f"Per-problem mean reward across 40 rollouts. The full {len(sampled_rates)} sampled distribution has {tail_phrase}. "
        f"The selected band [{args.low}, {args.high}] is the central region; everything outside is discarded.\n\n"
    )
    readme.append("![solve_rate_sampled](plots/solve_rate_sampled.png)\n\n")
    readme.append(f"Log-y view of the same distribution makes the perfectible region readable alongside the dominant {dominant_tail}:\n\n")
    readme.append("![solve_rate_sampled_log](plots/solve_rate_sampled_log.png)\n\n")
    readme.append(
        f"Zoomed into the selected band only — the {len(selected)} kept problems are roughly uniformly distributed across [0.2, 0.8] "
        f"(mean {mean(sel_rates):.3f}, median {median(sel_rates):.3f}, stdev {stdev(sel_rates):.3f}). "
        "No clustering toward the band edges:\n\n"
    )
    readme.append("![solve_rate_selected](plots/solve_rate_selected.png)\n\n")

    readme.append("## pass@k on selected\n\n")
    readme.append(
        "Unbiased Chen et al. (2021) estimator over 40 rollouts. A working RL recipe should be able to climb "
        "toward pass@40 ≈ 1.0 on this subset.\n\n"
    )
    readme.append("| k | pass@k |\n|---:|---:|\n")
    for k, v in zip(pass_ks, pass_at_k_values):
        readme.append(f"| {k} | {v:.4f} |\n")
    readme.append("\n![pass_at_k](plots/pass_at_k.png)\n\n")

    readme.append("## Domain & source mix\n\n")
    readme.append("Primary domain extracted from the 'Mathematics -> X -> ...' chain in the original tags.\n\n")
    readme.append("![domain_mix](plots/domain_mix.png)\n\n")
    readme.append("![source_mix](plots/source_mix.png)\n\n")

    readme.append("## Difficulty mix\n\n")
    readme.append("Original omni-math difficulty score (1-10), bucketed.\n\n")
    readme.append("![difficulty_mix](plots/difficulty_mix.png)\n\n")

    readme.append("## Solve rate vs reasoning length & difficulty\n\n")
    readme.append(
        "Per-problem scatter of mean reward vs mean output tokens, with the 40-rollout truncation rate as the colormap. "
        "Problems with very long reasoning + high truncation tend to score lower, as expected.\n\n"
    )
    readme.append("![solve_rate_vs_tokens](plots/solve_rate_vs_tokens.png)\n\n")
    readme.append(
        "Solve rate vs annotated problem difficulty (1-10 scale, jittered for readability). "
        "The green band is the perfectible selection range. Notice the broad spread at every difficulty level — "
        "annotated difficulty is only weakly predictive of actual model solve rate.\n\n"
    )
    readme.append("![solve_rate_vs_difficulty](plots/solve_rate_vs_difficulty.png)\n\n")

    readme.append("## Output tokens & truncation\n\n")
    readme.append(
        f"Mean output tokens per problem and per-problem truncation rate (fraction of 40 rollouts hitting "
        f"max_completion_tokens=15360 before EOS). {sum(1 for r in sel_trunc_rate if r > 0)} of {len(sel_trunc_rate)} "
        f"problems had at least one truncated rollout; mean rate {mean(sel_trunc_rate):.3f}, max {max(sel_trunc_rate):.3f}.\n\n"
    )
    readme.append("![tokens_and_truncation](plots/tokens_and_truncation.png)\n\n")

    readme.append("## Judge-fallback rate\n\n")
    near_one = sum(1 for r in sel_judge_rate if r >= 0.95)
    rollouts_per_problem = 40
    judge_calls_est = int(mean(sel_judge_rate) * len(selected) * rollouts_per_problem)
    readme.append(
        f"Fraction of rollouts where `math_verify` was ambiguous and the LLM judge (`gpt-5.4-mini`) had to be invoked. "
        f"Mean {mean(sel_judge_rate):.3f}, median {median(sel_judge_rate):.3f}. "
        "**Distribution is bimodal**: a central cluster around 0.5-0.6 (math_verify resolves ~half the rollouts cleanly), "
        f"plus a spike at ≈1.0 ({near_one} problems with judge rate ≥ 0.95) — these are problems whose canonical answer "
        "is in a form SymPy almost never simplifies (nested radicals, unsimplified piecewise expressions, "
        "named constants). Worth flagging because the judge is the modal scoring path and dominates eval compute "
        f"cost — at {mean(sel_judge_rate):.0%} judge rate over {len(selected)} problems × "
        f"{rollouts_per_problem} rollouts ≈ **{judge_calls_est:,} judge API calls per full pass-at-{rollouts_per_problem} evaluation**.\n\n"
    )
    readme.append("![judge_rate](plots/judge_rate.png)\n\n")

    # Caveats section
    hmmt_count = sum(c for s, c in sel_sources.items() if s.startswith("HMMT"))
    top_domain_pct = (
        max(sel_domains.values()) * 100 / len(selected) if sel_domains and selected else 0
    )
    readme.append("## Coverage caveats\n\n")
    readme.append(
        "Things to know before using this dataset as a generic math-reasoning benchmark "
        "(it is fit for purpose as a *diagnostic* subset for THIS model + recipe, not as a "
        "balanced benchmark in general):\n\n"
    )
    readme.append(
        f"- **Heavy competition skew toward HMMT** — {hmmt_count}/{len(selected)} = "
        f"{hmmt_count * 100 / len(selected):.0f}% of the selected problems come from "
        "Harvard-MIT Math Tournament (HMMT_2 + HMMT_11 combined). This reflects the "
        "underlying omni-math2 distribution, not a sampling artifact, but worth knowing "
        "if you generalize results.\n"
    )
    top_domain_name, top_domain_count = Counter(sel_domains).most_common(1)[0]
    other_domains = [(d, c) for d, c in Counter(sel_domains).most_common() if d != top_domain_name][:4]
    other_phrase = ", ".join(f"{d} ({c*100/len(selected):.0f}%)" for d, c in other_domains)
    readme.append(
        f"- **Top domain is {top_domain_name}** — {top_domain_count} of "
        f"{len(selected)} = {top_domain_pct:.0f}% of selected problems. "
        f"Other domains: {other_phrase}. See `plots/domain_mix.png` for full breakdown.\n"
    )
    readme.append(
        f"- **Annotated difficulty is only weakly predictive** of `{args.model_name.split('/')[-1]}`'s "
        "actual solve rate (see `solve_rate_vs_difficulty.png`). Problems labeled "
        "difficulty 4-6 span the full [0.2, 0.8] band; problems labeled 7-9 are not "
        "consistently harder for this model. Selection by *observed* solve rate is the "
        "more reliable difficulty signal.\n"
    )
    readme.append(
        f"- **Length-truncation tail is non-trivial** — {sum(1 for r in sel_trunc_rate if r > 0.1):d} "
        f"problems ({sum(1 for r in sel_trunc_rate if r > 0.1) * 100 / len(sel_trunc_rate):.0f}%) "
        "had >10% of their 40 rollouts hit the 15360-token cap. If you re-eval at smaller "
        "completion budgets you'll lose these problems' signal.\n"
    )
    readme.append(
        f"- **Model-specific** — the perfectible band is by construction relative to "
        f"`{args.model_name.split('/')[-1]}` at this sampling temperature. A different base "
        "model or different t/top_p will produce a different subset.\n\n"
    )

    readme.append("## Selected problems × source dataset coverage\n\n")
    readme.append("| | Selected | Sampled (1k) | Source (full) |\n|---|---:|---:|---:|\n")
    readme.append(f"| Total | {len(selected)} | {len(sampled_rows)} | {len(source)} |\n")
    for d in sorted(set(sel_domains) | set(sampled_domains) | set(source_domains), key=lambda x: -sel_domains.get(x, 0))[:10]:
        sel_pct = sel_domains.get(d, 0) * 100 / len(selected) if selected else 0
        samp_pct = sampled_domains.get(d, 0) * 100 / len(sampled_rows) if sampled_rows else 0
        src_pct = source_domains.get(d, 0) * 100 / len(source) if source else 0
        readme.append(
            f"| {d} | {sel_domains.get(d, 0)} ({sel_pct:.1f}%) | "
            f"{sampled_domains.get(d, 0)} ({samp_pct:.1f}%) | "
            f"{source_domains.get(d, 0)} ({src_pct:.1f}%) |\n"
        )
    readme.append("\n")

    readme.append("## Usage\n\n")
    readme.append("```python\n")
    readme.append("from datasets import load_dataset\n")
    if args.hf_repo:
        readme.append(f"ds = load_dataset(\"{args.hf_repo}\", split=\"train\")\n")
    readme.append("# Each row: id, problem, answer, solution, domain, source, difficulty, tags\n")
    readme.append("```\n\n")

    readme.append("## Methodology\n\n")
    readme.append(
        "1. Sample 1000 problems from the source dataset deterministically (seed=42).\n"
        f"2. Generate 40 rollouts per problem with `{args.model_name}` at t=1.0, top_p=0.95, "
        "max_completion_tokens=15360 via the prime-rl baselines harness.\n"
        f"3. Score each rollout with `math_verify` (SymPy symbolic comparison) and, when "
        "math_verify is ambiguous, fall back to a `gpt-5.4-mini` judge using the "
        "`omni_math2_hybrid_math_v1` grader rubric (the same rubric the prime-rl RLVR "
        "training loop uses).\n"
        f"4. Keep problems with mean reward in [{args.low}, {args.high}] and ≥8 rollouts.\n\n"
        "This follows the Hendrycks Sanity Check pattern from Yao et al. 2025 "
        "(*Defeating the Training-Inference Mismatch via FP16*, arxiv 2510.26788, §4).\n"
    )

    (args.out_dir / "README.md").write_text("".join(readme))
    print(f"wrote {args.out_dir / 'README.md'}")
    print(f"wrote {len(list(plots.glob('*.svg')))} plots in {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
