from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals.analyze_baseline_matrix import (
    PlotBox,
    compact_float,
    draw_axes,
    esc,
    nice_domain,
    red_green,
    safe_float,
    scale,
    summarize_run,
    svg_doc,
    write_csv,
)

KS = (1, 3, 5, 8, 16)

MODEL_LABELS = {
    "Qwen/Qwen3.5-4B": "Qwen3.5 4B",
    "Qwen/Qwen3.5-9B": "Qwen3.5 9B",
    "Qwen/Qwen3.5-27B": "Qwen3.5 27B",
    "Qwen/Qwen3.5-35B-A3B": "Qwen3.5 35B-A3B",
    "google/gemma-4-E4B-it": "Gemma4 E4B-it",
    "google/gemma-4-26B-A4B-it": "Gemma4 26B-A4B-it",
    "google/gemma-4-31B-it": "Gemma4 31B-it",
    "arcee-ai/Trinity-Mini": "Trinity Mini",
    "EssentialAI/rnj-1-instruct": "RNJ-1",
    "allenai/Olmo-3-7B-Instruct-SFT": "OLMo3 Inst SFT",
    "allenai/Olmo-3-7B-Instruct-DPO": "OLMo3 Inst DPO",
    "allenai/Olmo-3-7B-Think-SFT": "OLMo3 Think SFT",
    "allenai/Olmo-3-7B-Think-DPO": "OLMo3 Think DPO",
    "marin-community/marin-8b-instruct": "Marin 8B",
}

CANONICAL_RUNS = {
    "full-k16-v6-qwen35-4b-omni_math2-k16",
    "full-k16-v10-small-qwen35-9b-omni_math2-k16",
    "full-k16-v9-qwen35-27b-omni_math2-k16",
    "full-k16-v9-qwen35-35b-a3b-omni_math2-k16",
    "full-k16-v10-small-gemma4-e4b-it-omni_math2-k16",
    "full-k16-v12-gemma-nofreq-gemma4-26b-a4b-it-omni_math2-k16",
    "full-k16-v12-gemma-nofreq-gemma4-31b-it-omni_math2-k16",
    "full-k16-v19-trinity-merged-trinity-mini-omni_math2-k16",
    "full-k16-v11-longdense32-rnj-1-instruct-omni_math2-k16",
    "full-k16-v11-longdense32-olmo3-7b-instruct-sft-omni_math2-k16",
    "full-k16-v11-longdense32-olmo3-7b-instruct-dpo-omni_math2-k16",
    "full-k16-v11-longdense32-olmo3-7b-think-sft-omni_math2-k16",
    "full-k16-v11-longdense32-olmo3-7b-think-dpo-omni_math2-k16",
    "full-k16-v10-small-marin-8b-instruct-omni_math2-k16",
}

FAMILY_ORDER = {"Qwen3.5": 0, "Gemma4": 1, "Trinity": 2, "RNJ": 3, "OLMo3": 4, "Marin": 5}

PALETTE_14 = [
    "#2563eb",  # Qwen3.5 4B    — blue
    "#3b82f6",  # Qwen3.5 9B    — lighter blue
    "#1d4ed8",  # Qwen3.5 27B   — darker blue
    "#60a5fa",  # Qwen3.5 35B   — sky blue
    "#dc2626",  # Gemma4 E4B    — red
    "#ef4444",  # Gemma4 26B    — lighter red
    "#b91c1c",  # Gemma4 31B    — darker red
    "#f59e0b",  # Trinity Mini  — amber
    "#8b5cf6",  # RNJ-1         — violet
    "#059669",  # OLMo3 Inst SFT — emerald
    "#10b981",  # OLMo3 Inst DPO — green
    "#0d9488",  # OLMo3 Think SFT — teal
    "#14b8a6",  # OLMo3 Think DPO — lighter teal
    "#6b7280",  # Marin 8B      — gray
]


def model_label(row: dict[str, Any]) -> str:
    return MODEL_LABELS.get(str(row["model"]), str(row["model_short"]))


def model_family(model: str) -> str:
    m = model.lower()
    if "qwen" in m:
        return "Qwen3.5"
    if "gemma" in m:
        return "Gemma4"
    if "trinity" in m:
        return "Trinity"
    if "rnj" in m:
        return "RNJ"
    if "olmo" in m:
        return "OLMo3"
    if "marin" in m:
        return "Marin"
    return "Other"


def model_size(model: str) -> float:
    sizes = {
        "4B": 4.0, "E4B": 4.5, "9B": 9.0, "26B": 26.0,
        "27B": 27.0, "31B": 31.0, "35B": 35.0, "7B": 7.0, "8B": 8.0,
    }
    for marker, value in sizes.items():
        if marker in model:
            return value
    return 0.0


def model_variant_order(model: str) -> int:
    m = model.lower()
    if "think-dpo" in m or "think_dpo" in m:
        return 3
    if "think-sft" in m or "think_sft" in m:
        return 2
    if "instruct-dpo" in m or "instruct_dpo" in m:
        return 1
    return 0


def run_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    model = str(row["model"])
    fam = model_family(model)
    return (
        FAMILY_ORDER.get(fam, 99),
        model_size(model),
        model_variant_order(model),
        model,
    )


def color_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(rows, key=run_sort_key)
    labels = [model_label(row) for row in ordered]
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    assert len(seen) <= len(PALETTE_14), f"Need {len(seen)} colors but palette only has {len(PALETTE_14)}"
    return {label: PALETTE_14[i] for i, label in enumerate(seen)}


def fmt_pass(row: dict[str, Any], k: int) -> str:
    val = row.get(f"pass_at_{k}")
    if val is None:
        return "—"
    return f"{safe_float(val):.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| model | sample acc | pass@1 | pass@3 | pass@5 | pass@8 | pass@16 | pass16-pass1 | error | trunc | out tok mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pass1 = safe_float(row.get("pass_at_1"))
        pass16 = safe_float(row.get("pass_at_16"))
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(row),
                    f"{safe_float(row.get('sample_accuracy')):.3f}",
                    fmt_pass(row, 1),
                    fmt_pass(row, 3),
                    fmt_pass(row, 5),
                    fmt_pass(row, 8),
                    fmt_pass(row, 16),
                    f"{pass16 - pass1:.3f}",
                    f"{safe_float(row.get('error_rate')):.3f}",
                    f"{safe_float(row.get('truncation_rate')):.3f}",
                    f"{safe_float(row.get('output_tokens_mean')):.0f}",
                ]
            )
            + " |"
        )
    return lines


# ---------------------------------------------------------------------------
# SVG charts — adapted from analyze_baseline_matrix.py for single-task,
# 14-model Omni-Math2 setup
# ---------------------------------------------------------------------------


def write_pass_curve(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1080, 720
    box = PlotBox(78, 92, 680, 500)
    x_domain = (0, len(KS) - 1)
    y_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">Omni-MATH-2 pass@k curves</text>',
        '<text class="subtitle" x="34" y="64">Unbiased Chen-style estimator over k draws from 16 completions per problem (n=600).</text>',
        draw_axes(box, "k sampled attempts", "pass@k", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for i, k in enumerate(KS):
        x = scale(i, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{k}</text>')

    for row in sorted(rows, key=run_sort_key):
        points = []
        for i, k in enumerate(KS):
            val = row.get(f"pass_at_{k}")
            if val is None:
                continue
            px = scale(i, x_domain, (box.x, box.x + box.w))
            py = scale(safe_float(val), y_domain, (box.y + box.h, box.y))
            points.append((px, py))
        label = model_label(row)
        color = colors[label]
        if len(points) >= 2:
            point_s = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{point_s}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" stroke="#ffffff" stroke-width="1"/>')

    legend_x, legend_y = 785, 92
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y - 18}">Models</text>')
    for idx, row in enumerate(sorted(rows, key=run_sort_key)):
        y = legend_y + idx * 22
        label = model_label(row)
        color = colors[label]
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 30}" y="{y + 4}">{esc(label)}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_accuracy_cost(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1120, 670
    box = PlotBox(88, 94, 720, 470)
    xs = [safe_float(row["output_tokens_mean"]) for row in rows]
    x_domain = nice_domain(xs, floor=0)
    y_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">Accuracy vs generation cost</text>',
        '<text class="subtitle" x="34" y="64">Larger dark ring ∝ truncation rate. Labels are model abbreviations.</text>',
        draw_axes(box, "mean output tokens", "sample accuracy", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for tick in [0, 2048, 4096, 6144, 8192]:
        if x_domain[0] <= tick <= x_domain[1]:
            x = scale(tick, x_domain, (box.x, box.x + box.w))
            parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
            parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick}</text>')
    for row in sorted(rows, key=run_sort_key):
        x = scale(safe_float(row["output_tokens_mean"]), x_domain, (box.x, box.x + box.w))
        y = scale(safe_float(row["sample_accuracy"]), y_domain, (box.y + box.h, box.y))
        label = model_label(row)
        color = colors[label]
        ring = 5 + 14 * safe_float(row["truncation_rate"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{ring:.1f}" fill="none" stroke="#1f2933" stroke-opacity="0.35" stroke-width="1.3"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.1" fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 8:.1f}">{esc(label)}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_pass16_vs_trunc(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1120, 670
    box = PlotBox(88, 94, 720, 470)
    max_trunc = max(safe_float(row["truncation_rate"]) for row in rows)
    x_domain = (0.0, max(0.10, max_trunc * 1.08))
    y_domain = (0.0, 1.0)
    x_ticks = [t / 10 for t in range(0, int(x_domain[1] * 10) + 2) if t / 10 <= x_domain[1]]
    parts = [
        '<text class="title" x="34" y="42">pass@16 vs truncation rate</text>',
        '<text class="subtitle" x="34" y="64">Models in the upper-left are both accurate and concise; lower-right are token-budget-limited.</text>',
        draw_axes(box, "truncation rate", "pass@16", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for tick in x_ticks:
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick:.1f}</text>')
    for row in sorted(rows, key=run_sort_key):
        x = scale(safe_float(row["truncation_rate"]), x_domain, (box.x, box.x + box.w))
        y = scale(safe_float(row.get("pass_at_16", 0)), y_domain, (box.y + box.h, box.y))
        label = model_label(row)
        color = colors[label]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.3" fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 8:.1f}">{esc(label)}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_truncation_gap(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    ordered = sorted(rows, key=run_sort_key)
    width = 1120
    row_h = 32
    height = 120 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    x_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">Accuracy split by truncation</text>',
        '<text class="subtitle" x="34" y="64">Open dot = truncated completions; filled = non-truncated. Long lines → token cap is distorting measured capability.</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y - 10}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 22}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<line class="axis" x1="{box.x}" y1="{box.y + box.h}" x2="{box.x + box.w}" y2="{box.y + box.h}"/>')
    for i, row in enumerate(ordered):
        y = box.y + i * row_h + row_h / 2
        a_t = row.get("accuracy_truncated")
        a_n = row.get("accuracy_not_truncated")
        label = model_label(row)
        color = colors[label]
        parts.append(f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        if a_t in (None, "") or a_n in (None, ""):
            parts.append(f'<circle cx="{scale(safe_float(row.get("sample_accuracy", 0)), x_domain, (box.x, box.x + box.w)):.1f}" cy="{y:.1f}" r="4.8" fill="{color}" stroke="#fff" stroke-width="1"/>')
            continue
        x_t = scale(safe_float(a_t), x_domain, (box.x, box.x + box.w))
        x_n = scale(safe_float(a_n), x_domain, (box.x, box.x + box.w))
        parts.append(f'<line x1="{x_t}" y1="{y}" x2="{x_n}" y2="{y}" stroke="{color}" stroke-opacity="0.55" stroke-width="2"/>')
        parts.append(f'<circle cx="{x_t}" cy="{y}" r="4.2" fill="#fbfcfd" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x_n}" cy="{y}" r="4.8" fill="{color}" stroke="#fff" stroke-width="1"/>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_token_quantiles(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    ordered = sorted(rows, key=run_sort_key)
    width = 1120
    row_h = 32
    height = 120 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    max_tok = max(safe_float(row.get("output_tokens_p99", 0)) for row in ordered)
    x_cap = max(8192, math.ceil(max_tok / 2048) * 2048)
    x_domain = (0.0, float(x_cap))
    parts = [
        '<text class="title" x="34" y="42">Output-token distribution</text>',
        '<text class="subtitle" x="34" y="64">Line spans p50→p99, filled dot is p90. Pile-up at the right edge = truncation pressure.</text>',
    ]
    for tick in range(0, x_cap + 1, 2048):
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y - 10}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 22}" text-anchor="middle">{tick}</text>')
    for i, row in enumerate(ordered):
        y = box.y + i * row_h + row_h / 2
        p50 = safe_float(row.get("output_tokens_p50", 0))
        p90 = safe_float(row.get("output_tokens_p90", 0))
        p99 = safe_float(row.get("output_tokens_p99", 0))
        label = model_label(row)
        color = colors[label]
        x50 = scale(p50, x_domain, (box.x, box.x + box.w))
        x90 = scale(p90, x_domain, (box.x, box.x + box.w))
        x99 = scale(p99, x_domain, (box.x, box.x + box.w))
        parts.append(f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<line x1="{x50}" y1="{y}" x2="{x99}" y2="{y}" stroke="{color}" stroke-opacity="0.58" stroke-width="3"/>')
        parts.append(f'<circle cx="{x50}" cy="{y}" r="3.5" fill="#fbfcfd" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{x90}" cy="{y}" r="4.4" fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<circle cx="{x99}" cy="{y}" r="3.5" fill="#fbfcfd" stroke="{color}" stroke-width="1.5"/>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_pathology_heatmap(rows: list[dict[str, Any]], out: Path) -> None:
    ordered = sorted(rows, key=run_sort_key)
    cols = [
        ("truncation_rate", "trunc"),
        ("answer_tag_missing_when_truncated", "trunc no tag"),
        ("multi_answer_tag_rate", "multi tag"),
        ("loop_rate_truncated", "trunc loop"),
        ("empty_response_rate", "empty"),
        ("scaffold_leak_rate", "scaffold"),
        ("error_rate", "error"),
    ]
    width = 1120
    cell_w = 106
    row_h = 28
    x0, y0 = 245, 94
    height = 150 + row_h * len(ordered)
    parts = [
        '<text class="title" x="34" y="42">Format/pathology heatmap</text>',
        '<text class="subtitle" x="34" y="64">Darker red = more frequent. Separates capability from protocol/format failure.</text>',
    ]
    for j, (_, label) in enumerate(cols):
        x = x0 + j * cell_w + cell_w / 2
        parts.append(f'<text class="small" x="{x}" y="{y0 - 14}" text-anchor="middle">{esc(label)}</text>')
    for i, row in enumerate(ordered):
        y = y0 + i * row_h
        label = model_label(row)
        parts.append(f'<text class="small" x="{x0 - 12}" y="{y + 18}" text-anchor="end">{esc(label)}</text>')
        for j, (key, _) in enumerate(cols):
            value = safe_float(row.get(key))
            color = red_green(value)
            x = x0 + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{row_h - 4}" rx="3" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text class="small" x="{x + cell_w / 2 - 2}" y="{y + 17}" text-anchor="middle">{value:.2f}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def pareto_frontier(rows: list[dict[str, Any]], x_key: str, y_key: str) -> set[str]:
    frontier: set[str] = set()
    for row in rows:
        x = safe_float(row[x_key])
        y = safe_float(row[y_key])
        dominated = False
        for other in rows:
            if other is row:
                continue
            ox = safe_float(other[x_key])
            oy = safe_float(other[y_key])
            if ox <= x and oy >= y and (ox < x or oy > y):
                dominated = True
                break
        if not dominated:
            frontier.add(row["run_name"])
    return frontier


def write_pass16_cost_frontier(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1120, 670
    box = PlotBox(88, 94, 720, 470)
    xs = [safe_float(row["output_tokens_mean"]) for row in rows]
    x_domain = nice_domain(xs, floor=0)
    y_domain = (0.0, 1.0)
    frontier = pareto_frontier(rows, "output_tokens_mean", "pass_at_16")
    parts = [
        '<text class="title" x="34" y="42">pass@16 efficiency frontier</text>',
        '<text class="subtitle" x="34" y="64">Pareto-optimal models (bold ring) are not dominated on both cost and accuracy.</text>',
        draw_axes(box, "mean output tokens", "pass@16", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for tick in [0, 2048, 4096, 6144, 8192]:
        if x_domain[0] <= tick <= x_domain[1]:
            x = scale(tick, x_domain, (box.x, box.x + box.w))
            parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
            parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick}</text>')
    frontier_points = sorted(
        [row for row in rows if row["run_name"] in frontier],
        key=lambda row: safe_float(row["output_tokens_mean"]),
    )
    if len(frontier_points) >= 2:
        point_s = " ".join(
            f"{scale(safe_float(row['output_tokens_mean']), x_domain, (box.x, box.x + box.w)):.1f},"
            f"{scale(safe_float(row['pass_at_16']), y_domain, (box.y + box.h, box.y)):.1f}"
            for row in frontier_points
        )
        parts.append(f'<polyline points="{point_s}" fill="none" stroke="#111827" stroke-width="2.5" stroke-opacity="0.55"/>')
    for row in sorted(rows, key=run_sort_key):
        x = scale(safe_float(row["output_tokens_mean"]), x_domain, (box.x, box.x + box.w))
        y = scale(safe_float(row.get("pass_at_16", 0)), y_domain, (box.y + box.h, box.y))
        label = model_label(row)
        color = colors[label]
        is_frontier = row["run_name"] in frontier
        r = 7.4 if is_frontier else 5.2
        stroke = "#111827" if is_frontier else "#ffffff"
        stroke_w = 2.0 if is_frontier else 1.0
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_w}"/>')
        parts.append(f'<text class="small" x="{x + 9:.1f}" y="{y - 8:.1f}">{esc(label)}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


# ---------------------------------------------------------------------------
# Experimental analysis figures
# ---------------------------------------------------------------------------


def write_marginal_gain(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    """Δ pass@k per additional sample — shows diminishing returns and truncation ceilings."""
    width, height = 1080, 720
    box = PlotBox(78, 92, 680, 500)
    pairs = list(zip(KS[:-1], KS[1:]))
    x_domain = (0, len(pairs) - 1)
    all_deltas = []
    row_deltas: list[tuple[dict[str, Any], list[tuple[int, float]]]] = []
    for row in rows:
        deltas: list[tuple[int, float]] = []
        for idx, (k0, k1) in enumerate(pairs):
            v0 = row.get(f"pass_at_{k0}")
            v1 = row.get(f"pass_at_{k1}")
            if v0 is None or v1 is None:
                continue
            d = (safe_float(v1) - safe_float(v0)) / max(k1 - k0, 1)
            deltas.append((idx, d))
            all_deltas.append(d)
        row_deltas.append((row, deltas))

    y_domain = nice_domain(all_deltas, floor=0)
    y_ticks_raw = [i * 0.005 for i in range(0, int(y_domain[1] / 0.005) + 2)]
    y_ticks = [t for t in y_ticks_raw if t <= y_domain[1]]
    parts = [
        '<text class="title" x="34" y="42">Marginal gain per additional sample</text>',
        '<text class="subtitle" x="34" y="64">Δ pass@k / Δk between adjacent checkpoints. Flat lines = sampling ceiling reached; steep = headroom remains.</text>',
        draw_axes(box, "k interval", "Δ pass@k per sample", y_ticks, y_domain),
    ]
    x_labels = [f"{k0}→{k1}" for k0, k1 in pairs]
    for i, label in enumerate(x_labels):
        x = scale(i, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{label}</text>')

    for row, deltas in sorted(row_deltas, key=lambda rd: run_sort_key(rd[0])):
        points = [
            (scale(idx, x_domain, (box.x, box.x + box.w)), scale(d, y_domain, (box.y + box.h, box.y)))
            for idx, d in deltas
        ]
        label = model_label(row)
        color = colors[label]
        point_s = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_s}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')

    legend_x, legend_y = 785, 92
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y - 18}">Models</text>')
    seen: set[str] = set()
    idx = 0
    for row, _ in sorted(row_deltas, key=lambda rd: run_sort_key(rd[0])):
        label = model_label(row)
        if label in seen:
            continue
        seen.add(label)
        y = legend_y + idx * 22
        color = colors[label]
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 30}" y="{y + 4}">{esc(label)}</text>')
        idx += 1

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_cost_efficiency(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    """pass@1 per 1M output tokens — who gives you the most correct answers per token?"""
    width, height = 1120, 670
    box = PlotBox(88, 94, 720, 470)
    efficiencies = []
    for row in rows:
        p1 = safe_float(row.get("pass_at_1"))
        avg_tok = safe_float(row.get("output_tokens_mean"))
        eff = p1 / avg_tok * 1e6 if avg_tok > 0 else 0
        efficiencies.append((row, eff))

    ordered = sorted(efficiencies, key=lambda re: re[1], reverse=True)
    row_h = 36
    height = 130 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    max_eff = max(e for _, e in ordered)
    x_domain = (0.0, max_eff * 1.08)
    x_ticks = [i * 50 for i in range(0, int(x_domain[1] / 50) + 2) if i * 50 <= x_domain[1]]

    parts = [
        '<text class="title" x="34" y="42">Cost efficiency: correct answers per 1M output tokens</text>',
        '<text class="subtitle" x="34" y="64">pass@1 / avg_output_tokens × 10⁶. Higher = more bang for your token budget.</text>',
    ]
    for tick in x_ticks:
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y - 10}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 22}" text-anchor="middle">{tick:.0f}</text>')
    parts.append(f'<line class="axis" x1="{box.x}" y1="{box.y + box.h}" x2="{box.x + box.w}" y2="{box.y + box.h}"/>')

    for i, (row, eff) in enumerate(ordered):
        y = box.y + i * row_h + row_h / 2
        label = model_label(row)
        color = colors[label]
        x = scale(eff, x_domain, (box.x, box.x + box.w))
        bar_h = row_h * 0.55
        parts.append(f'<rect x="{box.x}" y="{y - bar_h / 2:.1f}" width="{x - box.x:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}" fill-opacity="0.75"/>')
        parts.append(f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<text class="small" x="{x + 6:.1f}" y="{y + 4}">{eff:.1f}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_effective_sample_size(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    """Effective k from truncation: if t% are truncated, effective draws ≈ (1-t)*16."""
    ordered = sorted(rows, key=run_sort_key)
    row_h = 36
    width = 1120
    height = 130 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    x_domain = (0.0, 16.0)
    parts = [
        '<text class="title" x="34" y="42">Effective sample size after truncation</text>',
        '<text class="subtitle" x="34" y="64">Nominal k=16. Effective k = (1 − truncation_rate) × 16. Models with low effective k are ceiling-bound.</text>',
    ]
    for tick in [0, 4, 8, 12, 16]:
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y - 10}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 22}" text-anchor="middle">{tick}</text>')
    parts.append(f'<line class="axis" x1="{box.x}" y1="{box.y + box.h}" x2="{box.x + box.w}" y2="{box.y + box.h}"/>')

    for i, row in enumerate(ordered):
        y = box.y + i * row_h + row_h / 2
        trunc = safe_float(row.get("truncation_rate", 0))
        eff_k = (1 - trunc) * 16
        label = model_label(row)
        color = colors[label]
        x = scale(eff_k, x_domain, (box.x, box.x + box.w))
        bar_h = row_h * 0.55
        parts.append(f'<rect x="{box.x}" y="{y - bar_h / 2:.1f}" width="{x - box.x:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}" fill-opacity="0.75"/>')
        parts.append(f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<text class="small" x="{x + 6:.1f}" y="{y + 4}">{eff_k:.1f}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_headroom_scatter(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    """pass@1 vs (pass@16 − pass@1) — capability vs sampling headroom."""
    width, height = 1120, 670
    box = PlotBox(88, 94, 720, 470)
    headrooms = [safe_float(row.get("pass_at_16", 0)) - safe_float(row.get("pass_at_1", 0)) for row in rows]
    x_domain = nice_domain([safe_float(row.get("pass_at_1", 0)) for row in rows], floor=0, ceil=1)
    y_domain = nice_domain(headrooms, floor=0)
    y_ticks = [i * 0.05 for i in range(0, int(y_domain[1] / 0.05) + 2) if i * 0.05 <= y_domain[1]]

    parts = [
        '<text class="title" x="34" y="42">Capability vs sampling headroom</text>',
        '<text class="subtitle" x="34" y="64">x = pass@1 (greedy-like); y = pass@16 − pass@1 (gain from best-of-16). Upper-right = strong and diverse.</text>',
        draw_axes(box, "pass@1", "pass@16 − pass@1", y_ticks, y_domain),
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        if x_domain[0] <= tick <= x_domain[1]:
            x = scale(tick, x_domain, (box.x, box.x + box.w))
            parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
            parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick:.2f}</text>')

    for row in sorted(rows, key=run_sort_key):
        p1 = safe_float(row.get("pass_at_1", 0))
        p16 = safe_float(row.get("pass_at_16", 0))
        headroom = p16 - p1
        x = scale(p1, x_domain, (box.x, box.x + box.w))
        y = scale(headroom, y_domain, (box.y + box.h, box.y))
        label = model_label(row)
        color = colors[label]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.3" fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 8:.1f}">{esc(label)}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(rows: list[dict[str, Any]], out: Path) -> None:
    total_rollouts = sum(int(row["n_rollouts"]) for row in rows)
    total_out = sum(safe_float(row["output_tokens_total"]) for row in rows)
    best_pass16 = max(rows, key=lambda row: safe_float(row.get("pass_at_16")))
    best_accuracy = max(rows, key=lambda row: safe_float(row.get("sample_accuracy")))
    max_headroom = max(
        rows,
        key=lambda row: safe_float(row.get("pass_at_16")) - safe_float(row.get("pass_at_1")),
    )
    max_error_rate = max(safe_float(row.get("error_rate")) for row in rows)
    most_efficient = max(
        rows,
        key=lambda row: safe_float(row.get("pass_at_1")) / max(safe_float(row.get("output_tokens_mean")), 1),
    )
    warning_lines = []
    if max_error_rate > 0:
        warning_lines = [
            "Warning: at least one run has non-zero `error_rate`; inspect `records.jsonl` before treating pass@K as model-only behavior.",
            "",
        ]

    lines = [
        "# Omni-MATH-2 Baseline Matrix Analysis",
        "",
        "Definitions: `sample acc` is per-completion correctness. `pass@k` is the unbiased Chen-style estimator over k draws sampled from 16 completions. `pass16-pass1` is a crude headroom proxy, not an RL gain estimate.",
        "",
        f"Runs analyzed: {len(rows)}. Rollouts: {total_rollouts:,}. Output tokens: {int(total_out):,}.",
        "",
        "## Headline",
        "",
        f"- Best pass@16: {model_label(best_pass16)} at {safe_float(best_pass16.get('pass_at_16')):.3f}.",
        f"- Best sample accuracy: {model_label(best_accuracy)} at {safe_float(best_accuracy.get('sample_accuracy')):.3f}.",
        f"- Largest sampling headroom: {model_label(max_headroom)} with pass@16-pass@1 = {safe_float(max_headroom.get('pass_at_16')) - safe_float(max_headroom.get('pass_at_1')):.3f}.",
        f"- Most cost-efficient: {model_label(most_efficient)} ({safe_float(most_efficient.get('pass_at_1')) / max(safe_float(most_efficient.get('output_tokens_mean')), 1) * 1e6:.1f} correct/Mtok).",
        f"- Max error rate: {max_error_rate:.3f}.",
        "",
        *warning_lines,
        "## Pass@K",
        "",
        *markdown_table(rows),
        "",
        "## Files",
        "",
        "- `run_metrics.csv` / `run_metrics.json`: flat per-run table.",
        "- `REPORT.md`: this summary.",
        "- `pass_curves.svg`: pass@k curves for all models.",
        "- `accuracy_vs_cost.svg`: sample accuracy vs mean output tokens.",
        "- `pass16_cost_frontier.svg`: pass@16/token Pareto frontier.",
        "- `pass16_vs_truncation.svg`: pass@16 vs truncation rate.",
        "- `truncation_accuracy_gap.svg`: accuracy split by truncated vs non-truncated.",
        "- `token_quantiles.svg`: p50→p99 output token spread.",
        "- `format_pathology_heatmap.svg`: truncation, tag loss, loops, empty, scaffold, error.",
        "- `marginal_gain.svg`: Δ pass@k per additional sample between checkpoints.",
        "- `cost_efficiency.svg`: correct answers per 1M output tokens.",
        "- `effective_sample_size.svg`: effective k after truncation loss.",
        "- `headroom_scatter.svg`: pass@1 vs sampling headroom (pass@16 − pass@1).",
    ]
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Omni-MATH-2 baseline matrix outputs.")
    parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/baselines/omni-math2-k16-tuned-20260503"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir
    output_dir = args.output_dir or matrix_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        path.parent
        for path in matrix_dir.glob("*/summary.json")
        if (path.parent / "records.jsonl").exists() and path.parent.name in CANONICAL_RUNS
    )
    if not run_dirs:
        raise SystemExit(f"No canonical runs found under {matrix_dir}")
    rows = sorted([summarize_run(run_dir) for run_dir in run_dirs], key=run_sort_key)
    assert len(rows) == len(CANONICAL_RUNS), f"expected {len(CANONICAL_RUNS)} runs, got {len(rows)}"

    colors = color_map(rows)

    write_csv(output_dir / "run_metrics.csv", rows)
    (output_dir / "run_metrics.json").write_text(json.dumps(rows, indent=2, default=compact_float) + "\n")
    write_report(rows, output_dir / "REPORT.md")

    write_pass_curve(rows, output_dir / "pass_curves.svg", colors)
    write_accuracy_cost(rows, output_dir / "accuracy_vs_cost.svg", colors)
    write_pass16_cost_frontier(rows, output_dir / "pass16_cost_frontier.svg", colors)
    write_pass16_vs_trunc(rows, output_dir / "pass16_vs_truncation.svg", colors)
    write_truncation_gap(rows, output_dir / "truncation_accuracy_gap.svg", colors)
    write_token_quantiles(rows, output_dir / "token_quantiles.svg", colors)
    write_pathology_heatmap(rows, output_dir / "format_pathology_heatmap.svg")

    write_marginal_gain(rows, output_dir / "marginal_gain.svg", colors)
    write_cost_efficiency(rows, output_dir / "cost_efficiency.svg", colors)
    write_effective_sample_size(rows, output_dir / "effective_sample_size.svg", colors)
    write_headroom_scatter(rows, output_dir / "headroom_scatter.svg", colors)

    print(f"wrote {len(rows)} run summaries + 11 charts to {output_dir}")


if __name__ == "__main__":
    main()
