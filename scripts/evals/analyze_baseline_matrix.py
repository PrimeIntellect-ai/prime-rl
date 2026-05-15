from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

KS = [1, 3, 5, 8, 16]
ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
OPEN_ANSWER_RE = re.compile(r"<answer\b", re.IGNORECASE)
CLOSE_ANSWER_RE = re.compile(r"</answer>", re.IGNORECASE)


MODEL_COLORS = [
    "#1f77b4",
    "#17becf",
    "#2ca02c",
    "#bcbd22",
    "#9467bd",
    "#8c564b",
    "#d62728",
    "#ff7f0e",
    "#e377c2",
    "#7f7f7f",
]


@dataclass(frozen=True)
class PlotBox:
    x: float
    y: float
    w: float
    h: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def frac(num: int, den: int) -> float:
    return num / den if den else 0.0


def compact_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 6)
    return value


def task_label(dataset: str) -> str:
    if "open" in dataset:
        return "OE"
    if "mcq" in dataset or "diamond" in dataset:
        return "MCQ"
    return dataset


def model_short(model: str) -> str:
    replacements = {
        "Qwen/Qwen3.5-4B-Base": "Qwen3.5 4B Base",
        "Qwen/Qwen3.5-4B": "Qwen3.5 4B",
        "Qwen/Qwen3.5-9B-Base": "Qwen3.5 9B Base",
        "Qwen/Qwen3.5-9B": "Qwen3.5 9B",
        "Qwen/Qwen3.5-35B-A3B-Base": "Qwen3.5 35B-A3B Base",
        "Qwen/Qwen3.5-35B-A3B": "Qwen3.5 35B-A3B",
        "google/gemma-4-E4B": "Gemma4 E4B Base",
        "google/gemma-4-E4B-it": "Gemma4 E4B-it",
        "google/gemma-4-26B-A4B": "Gemma4 26B-A4B Base",
        "google/gemma-4-26B-A4B-it": "Gemma4 26B-A4B-it",
    }
    return replacements.get(model, model.split("/")[-1])


def model_family(model: str) -> str:
    if model.startswith("Qwen/"):
        return "Qwen3.5"
    if model.startswith("google/gemma"):
        return "Gemma4"
    return model.split("/")[0]


def model_kind(model: str) -> str:
    if model.endswith("-Base") or model in {"google/gemma-4-E4B", "google/gemma-4-26B-A4B"}:
        return "base"
    if model.endswith("-it"):
        return "it"
    return "instruct"


def model_size_order(model: str) -> float:
    for marker, value in [
        ("4B", 4.0),
        ("E4B", 8.0),
        ("9B", 9.0),
        ("26B", 26.0),
        ("35B", 35.0),
    ]:
        if marker in model:
            return value
    return 0.0


def run_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    family_order = {"Qwen3.5": 0, "Gemma4": 1}
    kind_order = {"base": 0, "it": 1, "instruct": 2}
    task_order = {"MCQ": 0, "OE": 1}
    return (
        family_order.get(row["family"], 99),
        model_size_order(row["model"]),
        kind_order.get(row["kind"], 99),
        task_order.get(row["task"], 99),
    )


def extract_text(row: dict[str, Any]) -> str:
    response = row.get("response")
    if isinstance(response, str):
        return response
    completion = row.get("completion")
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    return ""


def answer_tag_stats(text: str) -> dict[str, float | int | bool | None]:
    matches = list(ANSWER_RE.finditer(text))
    open_count = len(OPEN_ANSWER_RE.findall(text))
    close_count = len(CLOSE_ANSWER_RE.findall(text))
    if not text:
        return {
            "answer_tag_complete": False,
            "answer_tag_open_count": open_count,
            "answer_tag_close_count": close_count,
            "answer_tag_count": 0,
            "answer_tag_start_pct": None,
            "answer_tag_end_pct": None,
            "answer_tag_text_len": 0,
        }
    if not matches:
        return {
            "answer_tag_complete": False,
            "answer_tag_open_count": open_count,
            "answer_tag_close_count": close_count,
            "answer_tag_count": 0,
            "answer_tag_start_pct": None,
            "answer_tag_end_pct": None,
            "answer_tag_text_len": 0,
        }
    first = matches[0]
    last = matches[-1]
    return {
        "answer_tag_complete": True,
        "answer_tag_open_count": open_count,
        "answer_tag_close_count": close_count,
        "answer_tag_count": len(matches),
        "answer_tag_start_pct": 100.0 * first.start() / max(len(text), 1),
        "answer_tag_end_pct": 100.0 * last.end() / max(len(text), 1),
        "answer_tag_text_len": len(last.group(1).strip()),
    }


def looks_looped(text: str) -> bool:
    if not text:
        return False

    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines() if len(line.strip()) >= 24]
    if lines and Counter(lines).most_common(1)[0][1] >= 3:
        return True

    words = re.findall(r"\w+", text.lower())
    if len(words) < 250:
        return False
    # Cap the diagnostic. This is a pathology hint, not a theorem prover.
    sample = words[:1200] + words[-1200:] if len(words) > 2400 else words
    grams = (" ".join(sample[i : i + 10]) for i in range(0, len(sample) - 9, 3))
    return any(count >= 4 for _, count in Counter(grams).most_common(8))


def scaffold_leak(text: str) -> bool:
    needles = [
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "<eos>",
        "assistant\n",
        "user\n",
        "model\n",
    ]
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def pass_value(summary: dict[str, Any], k: int, key: str) -> float | None:
    try:
        return safe_float(summary["pass"][str(k)][key])
    except KeyError:
        return None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    records_path = run_dir / "records.jsonl"
    if not summary_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"missing summary or records in {run_dir}")

    summary = json.loads(summary_path.read_text())
    rows = load_jsonl(records_path)
    if not rows:
        raise ValueError(f"no records in {records_path}")

    model = str(rows[0].get("model") or summary.get("model") or "")
    dataset = str(rows[0].get("dataset") or run_dir.name)
    task = task_label(dataset)
    texts = [extract_text(row) for row in rows]
    tag_rows = [answer_tag_stats(text) for text in texts]

    correct = [row for row in rows if row.get("correct") is True]
    incorrect = [row for row in rows if row.get("correct") is False]
    trunc_rows = [row for row in rows if row.get("is_truncated") is True]
    non_trunc_rows = [row for row in rows if row.get("is_truncated") is not True]
    trunc_indices = {id(row) for row in trunc_rows}

    output_tokens = [safe_float(row.get("output_tokens")) for row in rows]
    input_tokens = [safe_float(row.get("input_tokens")) for row in rows]
    generation_ms = [safe_float(row.get("generation_ms")) for row in rows]
    scoring_ms = [safe_float(row.get("scoring_ms")) for row in rows]
    total_ms = [safe_float(row.get("total_ms")) for row in rows]

    complete_tag_flags = [bool(stats["answer_tag_complete"]) for stats in tag_rows]
    tag_start_pcts = [
        safe_float(stats["answer_tag_start_pct"]) for stats in tag_rows if stats["answer_tag_start_pct"] is not None
    ]
    tag_end_pcts = [
        safe_float(stats["answer_tag_end_pct"]) for stats in tag_rows if stats["answer_tag_end_pct"] is not None
    ]
    tag_counts = [safe_float(stats["answer_tag_count"]) for stats in tag_rows]
    tag_text_lens = [safe_float(stats["answer_tag_text_len"]) for stats in tag_rows if stats["answer_tag_complete"]]
    multi_tag_flags = [safe_float(stats["answer_tag_open_count"]) > 1 for stats in tag_rows]

    trunc_complete_tags = [complete for row, complete in zip(rows, complete_tag_flags) if id(row) in trunc_indices]
    trunc_tag_start_pcts = [
        safe_float(stats["answer_tag_start_pct"])
        for row, stats in zip(rows, tag_rows)
        if id(row) in trunc_indices and stats["answer_tag_start_pct"] is not None
    ]
    trunc_tag_end_pcts = [
        safe_float(stats["answer_tag_end_pct"])
        for row, stats in zip(rows, tag_rows)
        if id(row) in trunc_indices and stats["answer_tag_end_pct"] is not None
    ]

    loop_flags = [looks_looped(text) for text in texts]
    scaffold_flags = [scaffold_leak(text) for text in texts]
    empty_flags = [not text.strip() for text in texts]

    def accuracy(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        return frac(sum(1 for row in group if row.get("correct") is True), len(group))

    tokens_per_s = [
        1000.0 * output / gen for output, gen in zip(output_tokens, generation_ms) if output > 0 and gen > 0
    ]

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "model": model,
        "model_short": model_short(model),
        "family": model_family(model),
        "kind": model_kind(model),
        "dataset": dataset,
        "task": task,
        "n_examples": int(summary.get("num_examples") or len({r.get("example_id") for r in rows})),
        "n_rollouts": int(summary.get("num_rollouts") or len(rows)),
        "sample_accuracy": safe_float(summary.get("mean_sample_accuracy", frac(len(correct), len(rows)))),
        "single_shot_accuracy": safe_float(summary.get("single_shot_accuracy")),
        "correct_rows": len(correct),
        "error_rate": safe_float(
            summary.get("error_rate", frac(sum(1 for r in rows if r.get("error") not in (None, "")), len(rows)))
        ),
        "record_error_rate": frac(sum(1 for r in rows if r.get("error") not in (None, "")), len(rows)),
        "truncation_rate": frac(len(trunc_rows), len(rows)),
        "accuracy_truncated": accuracy(trunc_rows),
        "accuracy_not_truncated": accuracy(non_trunc_rows),
        "accuracy_trunc_gap": (accuracy(non_trunc_rows) or 0.0) - (accuracy(trunc_rows) or 0.0),
        "empty_response_rate": frac(sum(empty_flags), len(rows)),
        "scaffold_leak_rate": frac(sum(scaffold_flags), len(rows)),
        "loop_rate": frac(sum(loop_flags), len(rows)),
        "loop_rate_truncated": frac(
            sum(1 for row, flag in zip(rows, loop_flags) if id(row) in trunc_indices and flag), len(trunc_rows)
        ),
        "multi_answer_tag_rate": frac(sum(multi_tag_flags), len(rows)),
        "multi_answer_tag_rate_truncated": frac(
            sum(1 for row, flag in zip(rows, multi_tag_flags) if id(row) in trunc_indices and flag), len(trunc_rows)
        ),
        "answer_tag_complete_rate": frac(sum(complete_tag_flags), len(rows)),
        "answer_tag_complete_rate_truncated": frac(sum(trunc_complete_tags), len(trunc_complete_tags)),
        "answer_tag_missing_when_truncated": 1.0 - frac(sum(trunc_complete_tags), len(trunc_complete_tags)),
        "answer_tag_start_p50_pct": pct(tag_start_pcts, 0.50),
        "answer_tag_start_p90_pct": pct(tag_start_pcts, 0.90),
        "answer_tag_end_p50_pct": pct(tag_end_pcts, 0.50),
        "answer_tag_end_p90_pct": pct(tag_end_pcts, 0.90),
        "answer_tag_start_trunc_p50_pct": pct(trunc_tag_start_pcts, 0.50),
        "answer_tag_end_trunc_p50_pct": pct(trunc_tag_end_pcts, 0.50),
        "answer_tag_count_mean": avg(tag_counts),
        "answer_tag_text_len_p50": pct(tag_text_lens, 0.50),
        "input_tokens_mean": avg(input_tokens),
        "input_tokens_total": sum(input_tokens),
        "output_tokens_mean": avg(output_tokens),
        "output_tokens_p50": pct(output_tokens, 0.50),
        "output_tokens_p75": pct(output_tokens, 0.75),
        "output_tokens_p90": pct(output_tokens, 0.90),
        "output_tokens_p95": pct(output_tokens, 0.95),
        "output_tokens_p99": pct(output_tokens, 0.99),
        "output_tokens_max": max(output_tokens) if output_tokens else None,
        "output_tokens_total": sum(output_tokens),
        "output_tokens_correct_mean": avg([safe_float(row.get("output_tokens")) for row in correct]),
        "output_tokens_incorrect_mean": avg([safe_float(row.get("output_tokens")) for row in incorrect]),
        "generation_ms_mean": avg(generation_ms),
        "scoring_ms_mean": avg(scoring_ms),
        "total_ms_mean": avg(total_ms),
        "tokens_per_second_mean": avg(tokens_per_s),
    }
    for k in KS:
        row[f"pass_at_{k}"] = pass_value(summary, k, "pass_at_k")
        row[f"pass_at_{k}_unbiased"] = pass_value(summary, k, "pass_at_k_unbiased_estimate")
        row[f"prefix_pass_at_{k}"] = pass_value(summary, k, "prefix_pass_at_k")
        row[f"successes_at_{k}_mean"] = pass_value(summary, k, "successes_at_k_mean")
        row[f"success_rate_at_{k}_mean"] = pass_value(summary, k, "success_rate_at_k_mean")
        row[f"all_pass_at_{k}"] = pass_value(summary, k, "all_pass_at_k")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: compact_float(row.get(key, "")) for key in keys})


def svg_doc(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }}
  .title {{ font-size: 21px; font-weight: 700; }}
  .subtitle {{ font-size: 12px; fill: #52616b; }}
  .axis {{ stroke: #9aa5b1; stroke-width: 1; }}
  .grid {{ stroke: #e4e7eb; stroke-width: 1; }}
  .tick {{ font-size: 11px; fill: #52616b; }}
  .label {{ font-size: 11px; fill: #334e68; }}
  .small {{ font-size: 10px; fill: #52616b; }}
</style>
<rect width="100%" height="100%" fill="#fbfcfd"/>
{body}
</svg>
"""


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    d0, d1 = domain
    r0, r1 = range_
    if d1 == d0:
        return (r0 + r1) / 2
    return r0 + (value - d0) * (r1 - r0) / (d1 - d0)


def nice_domain(values: list[float], floor: float | None = None, ceil: float | None = None) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals) if floor is None else floor
    hi = max(vals) if ceil is None else ceil
    if lo == hi:
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.06
    if floor is not None:
        lo = floor
    else:
        lo -= pad
    if ceil is not None:
        hi = ceil
    else:
        hi += pad
    return lo, hi


def color_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    models = sorted({row["model_short"] for row in rows}, key=lambda label: label.lower())
    return {model: MODEL_COLORS[i % len(MODEL_COLORS)] for i, model in enumerate(models)}


def draw_axes(box: PlotBox, x_label: str, y_label: str, y_ticks: list[float], y_domain: tuple[float, float]) -> str:
    parts = [
        f'<line class="axis" x1="{box.x}" y1="{box.y + box.h}" x2="{box.x + box.w}" y2="{box.y + box.h}"/>',
        f'<line class="axis" x1="{box.x}" y1="{box.y}" x2="{box.x}" y2="{box.y + box.h}"/>',
        f'<text class="label" x="{box.x + box.w / 2}" y="{box.y + box.h + 38}" text-anchor="middle">{esc(x_label)}</text>',
        f'<text class="label" transform="translate({box.x - 46},{box.y + box.h / 2}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>',
    ]
    for tick in y_ticks:
        y = scale(tick, y_domain, (box.y + box.h, box.y))
        parts.append(f'<line class="grid" x1="{box.x}" y1="{y}" x2="{box.x + box.w}" y2="{y}"/>')
        parts.append(f'<text class="tick" x="{box.x - 8}" y="{y + 4}" text-anchor="end">{tick:.1f}</text>')
    return "\n".join(parts)


def write_pass_curve(rows: list[dict[str, Any]], task: str, out: Path, colors: dict[str, str]) -> None:
    task_rows = [row for row in rows if row["task"] == task]
    width, height = 1080, 650
    box = PlotBox(78, 92, 720, 460)
    x_domain = (0, len(KS) - 1)
    y_domain = (0.0, 1.0)
    parts = [
        f'<text class="title" x="34" y="42">GPQA {esc(task)} pass@k curves</text>',
        '<text class="subtitle" x="34" y="64">pass@k is the unbiased estimator over k draws sampled from 16 completions; prefix-hit is not plotted here.</text>',
        draw_axes(box, "k sampled attempts", "pass@k", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for i, k in enumerate(KS):
        x = scale(i, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{k}</text>')

    for row in sorted(task_rows, key=run_sort_key):
        values = [safe_float(row.get(f"pass_at_{k}")) for k in KS]
        points = [
            (scale(i, x_domain, (box.x, box.x + box.w)), scale(v, y_domain, (box.y + box.h, box.y)))
            for i, v in enumerate(values)
        ]
        color = colors[row["model_short"]]
        point_s = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash = ' stroke-dasharray="5 4"' if row["kind"] == "base" else ""
        parts.append(f'<polyline points="{point_s}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>')
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.1" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )

    legend_x, legend_y = 825, 92
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y - 18}">Models</text>')
    for idx, row in enumerate(sorted(task_rows, key=run_sort_key)):
        y = legend_y + idx * 26
        color = colors[row["model_short"]]
        dash = ' stroke-dasharray="5 4"' if row["kind"] == "base" else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>'
        )
        parts.append(f'<text class="small" x="{legend_x + 32}" y="{y + 4}">{esc(row["model_short"])}</text>')

    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_accuracy_cost(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1120, 670
    box = PlotBox(88, 94, 760, 470)
    xs = [safe_float(row["output_tokens_mean"]) for row in rows]
    x_domain = nice_domain(xs, floor=0)
    y_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">Accuracy vs generation cost</text>',
        '<text class="subtitle" x="34" y="64">Circle = MCQ, square = open-ended. Point labels are model abbreviations; larger dark ring means more truncation.</text>',
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
        color = colors[row["model_short"]]
        ring = 5 + 12 * safe_float(row["truncation_rate"])
        if row["task"] == "MCQ":
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{ring:.1f}" fill="none" stroke="#1f2933" stroke-opacity="0.35" stroke-width="1.3"/>'
            )
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.1" fill="{color}" stroke="#fff" stroke-width="1"/>')
        else:
            s = 10.2
            parts.append(
                f'<rect x="{x - ring:.1f}" y="{y - ring:.1f}" width="{2 * ring:.1f}" height="{2 * ring:.1f}" fill="none" stroke="#1f2933" stroke-opacity="0.35" stroke-width="1.3"/>'
            )
            parts.append(
                f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>'
            )
        parts.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 8:.1f}">{esc(short_code(row))}</text>')

    legend_x, legend_y = 875, 100
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y - 22}">Reading</text>')
    parts.append(f'<circle cx="{legend_x + 8}" cy="{legend_y}" r="5" fill="#52616b"/>')
    parts.append(f'<text class="small" x="{legend_x + 24}" y="{legend_y + 4}">MCQ</text>')
    parts.append(f'<rect x="{legend_x + 2}" y="{legend_y + 22}" width="11" height="11" fill="#52616b"/>')
    parts.append(f'<text class="small" x="{legend_x + 24}" y="{legend_y + 32}">Open-ended</text>')
    parts.append(
        f'<circle cx="{legend_x + 8}" cy="{legend_y + 61}" r="12" fill="none" stroke="#1f2933" stroke-opacity="0.35"/>'
    )
    parts.append(f'<text class="small" x="{legend_x + 24}" y="{legend_y + 65}">ring size ∝ truncation</text>')
    out.write_text(svg_doc(width, height, "\n".join(parts)))


def short_code(row: dict[str, Any]) -> str:
    label = row["model_short"]
    label = label.replace("Qwen3.5 ", "Q").replace("Gemma4 ", "G")
    label = label.replace(" Base", "B").replace("-it", "it").replace("35B-A3B", "35A3")
    return f"{label}/{row['task']}"


def write_truncation_gap(rows: list[dict[str, Any]], out: Path) -> None:
    ordered = sorted(rows, key=run_sort_key)
    width = 1120
    row_h = 25
    height = 120 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    x_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">Accuracy split by truncation</text>',
        '<text class="subtitle" x="34" y="64">Open dot = truncated completions; filled dot = non-truncated completions. Long lines mean the token cap is changing the measured capability.</text>',
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
        if a_t in (None, "") or a_n in (None, ""):
            continue
        x_t = scale(safe_float(a_t), x_domain, (box.x, box.x + box.w))
        x_n = scale(safe_float(a_n), x_domain, (box.x, box.x + box.w))
        color = "#1f77b4" if row["family"] == "Qwen3.5" else "#d62728"
        parts.append(
            f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(short_code(row))}</text>'
        )
        parts.append(
            f'<line x1="{x_t}" y1="{y}" x2="{x_n}" y2="{y}" stroke="{color}" stroke-opacity="0.55" stroke-width="2"/>'
        )
        parts.append(f'<circle cx="{x_t}" cy="{y}" r="4.2" fill="#fbfcfd" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x_n}" cy="{y}" r="4.8" fill="{color}" stroke="#fff" stroke-width="1"/>')
    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_token_quantiles(rows: list[dict[str, Any]], out: Path) -> None:
    ordered = sorted(rows, key=run_sort_key)
    width = 1120
    row_h = 25
    height = 120 + row_h * len(ordered)
    box = PlotBox(245, 82, 760, row_h * len(ordered))
    x_domain = (0.0, 8192.0)
    parts = [
        '<text class="title" x="34" y="42">Output-token distribution</text>',
        '<text class="subtitle" x="34" y="64">Line spans p50→p99, filled dot is p90. A hard pile-up at 8192 is truncation pressure.</text>',
    ]
    for tick in [0, 2048, 4096, 6144, 8192]:
        x = scale(tick, x_domain, (box.x, box.x + box.w))
        parts.append(f'<line class="grid" x1="{x}" y1="{box.y - 10}" x2="{x}" y2="{box.y + box.h}"/>')
        parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 22}" text-anchor="middle">{tick}</text>')
    for i, row in enumerate(ordered):
        y = box.y + i * row_h + row_h / 2
        p50 = safe_float(row["output_tokens_p50"])
        p90 = safe_float(row["output_tokens_p90"])
        p99 = safe_float(row["output_tokens_p99"])
        color = "#1f77b4" if row["family"] == "Qwen3.5" else "#d62728"
        x50 = scale(p50, x_domain, (box.x, box.x + box.w))
        x90 = scale(p90, x_domain, (box.x, box.x + box.w))
        x99 = scale(p99, x_domain, (box.x, box.x + box.w))
        parts.append(
            f'<text class="small" x="{box.x - 10}" y="{y + 4}" text-anchor="end">{esc(short_code(row))}</text>'
        )
        parts.append(
            f'<line x1="{x50}" y1="{y}" x2="{x99}" y2="{y}" stroke="{color}" stroke-opacity="0.58" stroke-width="3"/>'
        )
        parts.append(f'<circle cx="{x50}" cy="{y}" r="3.5" fill="#fbfcfd" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{x90}" cy="{y}" r="4.4" fill="{color}" stroke="#fff" stroke-width="1"/>')
        parts.append(f'<circle cx="{x99}" cy="{y}" r="3.5" fill="#fbfcfd" stroke="{color}" stroke-width="1.5"/>')
    out.write_text(svg_doc(width, height, "\n".join(parts)))


def red_green(value: float, invert: bool = False) -> str:
    value = min(max(value, 0.0), 1.0)
    if invert:
        value = 1.0 - value
    # light yellow at 0.5, green at good, red at bad.
    if value <= 0.5:
        t = value / 0.5
        r = int(46 + t * (255 - 46))
        g = int(125 + t * (238 - 125))
        b = int(50 + t * (160 - 50))
    else:
        t = (value - 0.5) / 0.5
        r = int(255 + t * (198 - 255))
        g = int(238 + t * (40 - 238))
        b = int(160 + t * (40 - 160))
    return f"#{r:02x}{g:02x}{b:02x}"


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
    cell_w = 118
    row_h = 28
    x0, y0 = 300, 94
    height = 150 + row_h * len(ordered)
    parts = [
        '<text class="title" x="34" y="42">Format/pathology heatmap</text>',
        '<text class="subtitle" x="34" y="64">Darker red = more frequent. This separates capability from protocol failure.</text>',
    ]
    for j, (_, label) in enumerate(cols):
        x = x0 + j * cell_w + cell_w / 2
        parts.append(f'<text class="small" x="{x}" y="{y0 - 14}" text-anchor="middle">{esc(label)}</text>')
    for i, row in enumerate(ordered):
        y = y0 + i * row_h
        parts.append(f'<text class="small" x="{x0 - 12}" y="{y + 18}" text-anchor="end">{esc(short_code(row))}</text>')
        for j, (key, _) in enumerate(cols):
            value = safe_float(row.get(key))
            color = red_green(value)
            x = x0 + j * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{row_h - 4}" rx="3" fill="{color}" stroke="#ffffff"/>'
            )
            parts.append(
                f'<text class="small" x="{x + cell_w / 2 - 2}" y="{y + 17}" text-anchor="middle">{value:.2f}</text>'
            )
    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_pass16_vs_trunc(rows: list[dict[str, Any]], out: Path, colors: dict[str, str]) -> None:
    width, height = 1120, 670
    box = PlotBox(88, 94, 760, 470)
    x_domain = (0.0, max(0.55, max(safe_float(row["truncation_rate"]) for row in rows) * 1.08))
    y_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">pass@16 vs truncation</text>',
        '<text class="subtitle" x="34" y="64">This is the quickest view of “better because more samples” versus “broken by long generations.”</text>',
        draw_axes(box, "truncation rate", "pass@16", [0, 0.25, 0.5, 0.75, 1.0], y_domain),
    ]
    for tick in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        if tick <= x_domain[1]:
            x = scale(tick, x_domain, (box.x, box.x + box.w))
            parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
            parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick:.1f}</text>')
    for row in sorted(rows, key=run_sort_key):
        x = scale(safe_float(row["truncation_rate"]), x_domain, (box.x, box.x + box.w))
        y = scale(safe_float(row["pass_at_16"]), y_domain, (box.y + box.h, box.y))
        color = colors[row["model_short"]]
        if row["task"] == "MCQ":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.3" fill="{color}" stroke="#fff" stroke-width="1"/>')
        else:
            parts.append(
                f'<rect x="{x - 5.1:.1f}" y="{y - 5.1:.1f}" width="10.2" height="10.2" fill="{color}" stroke="#fff" stroke-width="1"/>'
            )
        parts.append(f'<text class="small" x="{x + 8:.1f}" y="{y - 8:.1f}">{esc(short_code(row))}</text>')
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
    panels = {
        "MCQ": PlotBox(88, 104, 450, 420),
        "OE": PlotBox(610, 104, 450, 420),
    }
    xs = [safe_float(row["output_tokens_mean"]) for row in rows]
    x_domain = nice_domain(xs, floor=0)
    y_domain = (0.0, 1.0)
    parts = [
        '<text class="title" x="34" y="42">pass@16 efficiency frontier</text>',
        '<text class="subtitle" x="34" y="64">Pareto points are not dominated by another model with both lower mean output tokens and higher pass@16.</text>',
    ]
    for task, box in panels.items():
        task_rows = [row for row in rows if row["task"] == task]
        frontier = pareto_frontier(task_rows, "output_tokens_mean", "pass_at_16")
        parts.append(f'<text class="label" x="{box.x}" y="{box.y - 20}">{task}</text>')
        parts.append(draw_axes(box, "mean output tokens", "pass@16", [0, 0.25, 0.5, 0.75, 1.0], y_domain))
        for tick in [0, 2048, 4096, 6144, 8192]:
            if x_domain[0] <= tick <= x_domain[1]:
                x = scale(tick, x_domain, (box.x, box.x + box.w))
                parts.append(f'<line class="grid" x1="{x}" y1="{box.y}" x2="{x}" y2="{box.y + box.h}"/>')
                parts.append(f'<text class="tick" x="{x}" y="{box.y + box.h + 20}" text-anchor="middle">{tick}</text>')
        frontier_points = sorted(
            [row for row in task_rows if row["run_name"] in frontier],
            key=lambda row: safe_float(row["output_tokens_mean"]),
        )
        if len(frontier_points) >= 2:
            point_s = " ".join(
                f"{scale(safe_float(row['output_tokens_mean']), x_domain, (box.x, box.x + box.w)):.1f},"
                f"{scale(safe_float(row['pass_at_16']), y_domain, (box.y + box.h, box.y)):.1f}"
                for row in frontier_points
            )
            parts.append(
                f'<polyline points="{point_s}" fill="none" stroke="#111827" stroke-width="2.5" stroke-opacity="0.55"/>'
            )
        for row in sorted(task_rows, key=run_sort_key):
            x = scale(safe_float(row["output_tokens_mean"]), x_domain, (box.x, box.x + box.w))
            y = scale(safe_float(row["pass_at_16"]), y_domain, (box.y + box.h, box.y))
            color = colors[row["model_short"]]
            is_frontier = row["run_name"] in frontier
            r = 7.4 if is_frontier else 5.2
            stroke = "#111827" if is_frontier else "#ffffff"
            stroke_w = 2.0 if is_frontier else 1.0
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
            )
            parts.append(
                f'<text class="small" x="{x + 9:.1f}" y="{y - 8:.1f}">{esc(short_code(row).replace("/" + task, ""))}</text>'
            )
    out.write_text(svg_doc(width, height, "\n".join(parts)))


def write_report(rows: list[dict[str, Any]], out: Path) -> None:
    def best(task: str, key: str) -> dict[str, Any]:
        task_rows = [row for row in rows if row["task"] == task]
        return max(task_rows, key=lambda row: safe_float(row.get(key)))

    def worst(task: str, key: str) -> dict[str, Any]:
        task_rows = [row for row in rows if row["task"] == task]
        return max(task_rows, key=lambda row: safe_float(row.get(key)))

    total_out = sum(safe_float(row["output_tokens_total"]) for row in rows)
    total_in = sum(safe_float(row["input_tokens_total"]) for row in rows)
    total_rollouts = sum(int(row["n_rollouts"]) for row in rows)

    lines = [
        "# GPQA Baseline Matrix Analysis",
        "",
        "Definitions: `sample_accuracy` is per-completion correctness. `single_shot_accuracy` is trial 0 only. `pass@k` is the unbiased estimator over k draws from 16 sampled completions, not prefix-hit.",
        "",
        f"Runs analyzed: {len(rows)}. Rollouts: {total_rollouts:,}. Input tokens: {int(total_in):,}. Output tokens: {int(total_out):,}.",
        "",
        "## Headline",
        "",
    ]
    for task in ["MCQ", "OE"]:
        b_pass = best(task, "pass_at_16")
        b_acc = best(task, "sample_accuracy")
        w_trunc = worst(task, "truncation_rate")
        lines.extend(
            [
                f"- {task}: best pass@16 is {b_pass['model_short']} at {safe_float(b_pass['pass_at_16']):.3f}; best sample accuracy is {b_acc['model_short']} at {safe_float(b_acc['sample_accuracy']):.3f}.",
                f"- {task}: worst truncation is {w_trunc['model_short']} at {safe_float(w_trunc['truncation_rate']):.3f}; its truncated accuracy is {fmt_optional(w_trunc.get('accuracy_truncated'))}, non-truncated accuracy is {fmt_optional(w_trunc.get('accuracy_not_truncated'))}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `run_metrics.csv` / `run_metrics.json`: flat per-run table.",
            "- `pass_curves_mcq.svg`, `pass_curves_oe.svg`: pass@k curves.",
            "- `accuracy_vs_cost.svg`: sample accuracy vs mean output tokens.",
            "- `pass16_cost_frontier.svg`: pass@16/token Pareto frontier.",
            "- `pass16_vs_truncation.svg`: pass@16 vs truncation.",
            "- `token_quantiles.svg`: p50→p99 output token spread.",
            "- `truncation_accuracy_gap.svg`: accuracy split by truncated vs non-truncated samples.",
            "- `format_pathology_heatmap.svg`: truncation, tag loss, loops, empty outputs, scaffolding, and errors.",
        ]
    )
    out.write_text("\n".join(lines) + "\n")


def fmt_optional(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{safe_float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPQA baseline matrix outputs.")
    parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/baselines/matrix-20260501"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir
    output_dir = args.output_dir or matrix_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        path.parent for path in matrix_dir.glob("*/summary.json") if (path.parent / "records.jsonl").exists()
    )
    rows = sorted([summarize_run(run_dir) for run_dir in run_dirs], key=run_sort_key)
    colors = color_map(rows)

    write_csv(output_dir / "run_metrics.csv", rows)
    (output_dir / "run_metrics.json").write_text(json.dumps(rows, indent=2, default=compact_float) + "\n")
    write_report(rows, output_dir / "REPORT.md")

    write_pass_curve(rows, "MCQ", output_dir / "pass_curves_mcq.svg", colors)
    write_pass_curve(rows, "OE", output_dir / "pass_curves_oe.svg", colors)
    write_accuracy_cost(rows, output_dir / "accuracy_vs_cost.svg", colors)
    write_pass16_cost_frontier(rows, output_dir / "pass16_cost_frontier.svg", colors)
    write_pass16_vs_trunc(rows, output_dir / "pass16_vs_truncation.svg", colors)
    write_truncation_gap(rows, output_dir / "truncation_accuracy_gap.svg")
    write_token_quantiles(rows, output_dir / "token_quantiles.svg")
    write_pathology_heatmap(rows, output_dir / "format_pathology_heatmap.svg")

    print(f"wrote {len(rows)} run summaries to {output_dir}")


if __name__ == "__main__":
    main()
