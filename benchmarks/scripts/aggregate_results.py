#!/usr/bin/env python3
"""Aggregate benchmark results from multiple JSON files into markdown report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseSettings, parse_argv

SHORTENED_ATTN_MAPPING = {
    "flash_attention_2": "FA2",
    "flash_attention_3": "FA3",
}

# These words are stripped from the device name to get the short name
DEVICE_NAME_STRIP_WORDS = ["NVIDIA", "RTX", "80GB", "40GB"]


class AggregateConfig(BaseSettings):
    """Configuration for aggregating benchmark results."""

    artifacts_dir: Annotated[Path, Field(description="Directory containing benchmark JSON artifacts")]
    baselines_dir: Annotated[Path, Field(description="Directory containing baseline JSON artifacts")]
    output_markdown: Annotated[Path, Field(description="Output markdown file path")]
    regression_threshold: Annotated[float, Field(description="Regression threshold (default: 0.05 = 5%)")] = 0.05
    diff_display_threshold: Annotated[float, Field(description="Diff display threshold (default: 0.01 = 1%)")] = 0.01


def get_hardware(config: dict) -> str:
    num_gpus = config.get("num_gpus", 1)
    device_name = config.get("device_name", "Unknown")
    short_name = " ".join(word for word in device_name.split() if word not in DEVICE_NAME_STRIP_WORDS)
    return f"{num_gpus}x{short_name}"


def get_training_type(config: dict) -> str:
    train_type = config.get("type", "unknown").upper()
    lora_rank = config.get("lora_rank")
    if lora_rank is not None:
        return f"{train_type} LoRA(r={lora_rank})"
    return f"{train_type} Full"


def get_config_key(config: dict) -> str:
    """Get a unique key for a benchmark configuration."""
    return "|".join(
        [
            config.get("model_name", "unknown"),
            get_hardware(config),
            config.get("type", "rl"),
            str(config.get("lora_rank") or "none"),
            str(config.get("seq_len", 0)),
            config.get("ac", "None"),
            config.get("attention", "unknown"),
        ]
    )


def load_json_dir(directory: Path) -> list[dict]:
    """Load all JSON files from a directory."""
    results = []
    if not directory.exists():
        return results
    for f in directory.rglob("*.json"):
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}", file=sys.stderr)
    return results


def format_metric(
    value: float,
    baseline: float | None,
    threshold: float,
    diff_display_threshold: float,
    fmt: str,
    higher_is_better: bool = True,
) -> tuple[str, bool]:
    formatted = fmt.format(value)
    if baseline is None or baseline == 0:
        return formatted, False

    pct_change = (value - baseline) / baseline * 100
    is_regression = (value < baseline * (1 - threshold)) if higher_is_better else (value > baseline * (1 + threshold))

    if is_regression:
        return f"**{formatted}** :warning: ({pct_change:+.1f}%)", True
    elif abs(pct_change) >= diff_display_threshold * 100:
        return f"{formatted} ({pct_change:+.1f}%)", False
    return formatted, False


def sorting_key(x: dict) -> tuple:
    return (
        get_training_type(x["config"]),
        x["config"]["seq_len"],
        -x["metrics"]["throughput"]["mean"],
        get_hardware(x["config"]),
        x["config"]["ac"],
        x["config"]["attention"],
    )


def generate_markdown(
    results: list[dict],
    baselines: dict[str, dict],
    threshold: float,
    diff_display_threshold: float,
) -> tuple[str, bool]:
    has_regressions = False

    by_model: dict[str, list[dict]] = {}
    for r in results:
        if r["config"]["success"]:
            model = r["config"]["model_name"]
            by_model.setdefault(model, []).append(r)

    commit = results[0]["config"].get("commit_sha", "unknown") if results else "unknown"
    docker_image = results[0]["config"].get("docker_image", "unknown") if results else "unknown"

    lines = [
        "# Performance Benchmarks",
        "",
        "Automated benchmark results for prime-rl using `--bench` flag.",
        "",
        f"**Last Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Commit:** `{commit}`  ",
        f"**Docker Image:** `{docker_image}`",
        "",
        f"> :warning: indicates regression > {threshold * 100:.0f}% from baseline",
        f"> diffs shown when abs(change) >= {diff_display_threshold * 100:.1f}% (except regressions, which always show diffs)",
        "",
        "> :clock10: The Step Time shown is the time taken per micro batch. This differs from what gets displayed in the bench table which is the total step time."
        "",
    ]

    for model, model_results in sorted(by_model.items()):
        lines.append(f"## {model.split('/')[-1]}")
        lines.append("")
        lines.append("| Type | SeqLen | AC | Attn | Hardware | MFU | TPS | Step Time | Peak Mem |")
        lines.append("|------|--------|----|----|----------|-----|-----|-----------|----------|")

        for r in sorted(model_results, key=sorting_key):
            cfg, m = r["config"], r["metrics"]
            bl = baselines.get(get_config_key(cfg), {})

            mfu_str, mfu_reg = format_metric(
                m["mfu"]["mean"], bl.get("mfu"), threshold, diff_display_threshold, "{:.1f}%", True
            )
            tps_str, tps_reg = format_metric(
                m["throughput"]["mean"] / 1000,
                bl.get("throughput", 0) / 1000 if bl.get("throughput") else None,
                threshold,
                diff_display_threshold,
                "{:.2f}k",
                True,
            )
            step_str, step_reg = format_metric(
                cfg["seq_len"] * cfg["num_gpus"] / m["throughput"]["mean"],
                None,
                threshold,
                diff_display_threshold,
                "{:.2f}s",
                False,
            )

            has_regressions = has_regressions or mfu_reg or tps_reg or step_reg
            attn = SHORTENED_ATTN_MAPPING.get(cfg["attention"], cfg["attention"])

            lines.append(
                f"| {get_training_type(cfg)} | {cfg['seq_len']} | {cfg.get('ac', 'None')} | {attn} | {get_hardware(cfg)} | "
                f"{mfu_str} | {tps_str} | {step_str} | {m['peak_memory']['gib']:.1f} GiB |"
            )
        lines.append("")

    failed = [r for r in results if not r["config"]["success"]]
    if failed:
        lines.append("## Failed Benchmarks")
        lines.append("")
        for r in failed:
            cfg = r["config"]
            error = (cfg.get("error_reason") or "Unknown error")[:200]
            lines.append(f"- **{cfg['model_name']}** ({get_training_type(cfg)}) on {get_hardware(cfg)}: {error}")
        lines.append("")

    return "\n".join(lines), has_regressions


def main():
    config = parse_argv(AggregateConfig)

    results = load_json_dir(config.artifacts_dir)
    print(f"Loaded {len(results)} benchmark results", file=sys.stderr)

    baseline_results = load_json_dir(config.baselines_dir)
    baselines = {}
    for r in baseline_results:
        if r["config"]["success"]:
            m = r["metrics"]
            baselines[get_config_key(r["config"])] = {
                "mfu": m["mfu"]["mean"],
                "throughput": m["throughput"]["mean"],
                "step_time": m["step_time"]["mean"],
            }
    print(f"Loaded {len(baselines)} baseline entries", file=sys.stderr)

    markdown, has_regressions = generate_markdown(
        results, baselines, config.regression_threshold, config.diff_display_threshold
    )
    config.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.output_markdown.write_text(markdown)
    print(f"Wrote markdown report to {config.output_markdown}", file=sys.stderr)

    successful = sum(1 for r in results if r["config"]["success"])
    print(f"\nSummary: {successful} successful, {len(results) - successful} failed benchmarks", file=sys.stderr)

    if has_regressions:
        Path("/tmp/prime-rl-benchmark-aggregate-has-regressions").touch()
        print("WARNING: Performance regressions detected!", file=sys.stderr)


if __name__ == "__main__":
    main()
