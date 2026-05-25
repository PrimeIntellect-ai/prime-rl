#!/usr/bin/env python3
"""Render a static HTML comparison of DPPO and IcePop token masks.

The input is a prime-rl token export JSONL file, a step directory, or the
token_exports root directory. By default, a directory input resolves to the
latest step_N directory.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None

try:
    import orjson
except ImportError:  # pragma: no cover - optional speedup
    orjson = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize DPPO vs IcePop masking on prime-rl token exports.",
    )
    parser.add_argument("input", type=Path, help="JSONL file, step directory, or token_exports root.")
    parser.add_argument("--output", "-o", type=Path, help="HTML output path.")
    parser.add_argument("--step", type=int, help="Step to render when input is a token_exports root.")
    parser.add_argument("--all-steps", action="store_true", help="Scan all steps under the input directory.")
    parser.add_argument("--rank", type=int, help="Only include one trainer rank.")
    parser.add_argument("--env-name", help="Only include one env_name.")
    parser.add_argument("--top-records", type=int, default=12, help="Number of disagreement-heavy records to embed.")
    parser.add_argument("--all-records", action="store_true", help="Embed every matched record.")
    parser.add_argument("--window-tokens", type=int, default=2200, help="Token window around each selected focus token.")
    parser.add_argument("--full-record", action="store_true", help="Embed full selected records instead of windows.")
    parser.add_argument("--tokenizer", help="Optional tokenizer name/path for token-id decoding.")
    parser.add_argument("--no-decode", action="store_true", help="Always show token ids instead of decoded text.")
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow transformers to download tokenizer files. By default only local files are used.",
    )
    parser.add_argument("--dppo-mask-low", type=float, help="DPPO low prob_delta threshold.")
    parser.add_argument("--dppo-mask-high", type=float, help="DPPO high prob_delta threshold.")
    parser.add_argument("--icepop-ratio-low", type=float, default=0.2, help="IcePop lower importance-ratio bound.")
    parser.add_argument("--icepop-ratio-high", type=float, default=5.0, help="IcePop upper importance-ratio bound.")
    parser.add_argument(
        "--max-mismatch",
        type=float,
        help="KL value mapped to deepest red. Defaults to p95 of embedded trainable tokens.",
    )
    args = parser.parse_args()

    if args.top_records <= 0:
        raise ValueError("--top-records must be positive.")
    if args.window_tokens <= 0 and not args.full_record:
        raise ValueError("--window-tokens must be positive unless --full-record is set.")

    files, root, step = resolve_files(args.input, step=args.step, all_steps=args.all_steps)
    thresholds = {
        "dppo_mask_low": resolve_dppo_threshold(args.input, "dppo_mask_low", args.dppo_mask_low),
        "dppo_mask_high": resolve_dppo_threshold(args.input, "dppo_mask_high", args.dppo_mask_high),
        "icepop_ratio_low": args.icepop_ratio_low,
        "icepop_ratio_high": args.icepop_ratio_high,
    }
    tokenizer = None
    if args.tokenizer and not args.no_decode:
        tokenizer = load_tokenizer(args.tokenizer, local_files_only=not args.allow_tokenizer_download)

    scan = empty_group()
    scan.update({"files": len(files), "records_scanned": 0, "records_matched": 0})
    examples: dict[str, list[tuple[tuple[float, int, int, int, int], dict[str, Any]]]] = {
        "icepop_only": [],
        "dppo_only": [],
        "both": [],
        "unmasked_high_mismatch": [],
    }
    selected: list[tuple[tuple[int, float, int, int], dict[str, Any]]] = []
    prepared: list[dict[str, Any]] = []
    sequence = 0

    for record, source, line_number in iter_records(files, root=root, rank=args.rank, env_name=args.env_name):
        scan["records_scanned"] += 1
        scan["records_matched"] += 1
        summary = summarize_record(record, thresholds)
        summary.update(
            {
                "source_file": source,
                "source_line": line_number,
                "scan_index": scan["records_matched"] - 1,
            },
        )
        merge_counts(scan, summary)
        collect_examples(examples, record, summary, thresholds)

        if args.all_records:
            prepared.append(
                prepare_record(
                    record,
                    summary=summary,
                    ordinal=len(prepared),
                    thresholds=thresholds,
                    tokenizer=tokenizer,
                    window_tokens=args.window_tokens,
                    full_record=args.full_record,
                ),
            )
        else:
            score = (
                int(summary["disagree_count"]),
                float(summary["max_disagree_mismatch"] or -math.inf),
                int(summary["trainable_tokens"]),
                sequence,
            )
            item = {"record": record, "summary": summary}
            if len(selected) < args.top_records:
                heapq.heappush(selected, (score, item))
            elif score > selected[0][0]:
                heapq.heapreplace(selected, (score, item))
        sequence += 1

    finalize_rates(scan)
    if not prepared:
        for _, item in sorted(selected, reverse=True):
            prepared.append(
                prepare_record(
                    item["record"],
                    summary=item["summary"],
                    ordinal=len(prepared),
                    thresholds=thresholds,
                    tokenizer=tokenizer,
                    window_tokens=args.window_tokens,
                    full_record=args.full_record,
                ),
            )
    if not prepared:
        raise ValueError("No token export records matched the requested filters.")

    top_examples = {
        name: [row for _, row in sorted(rows, reverse=True)]
        for name, rows in examples.items()
    }
    decode_examples(top_examples, tokenizer)

    output = args.output or default_output(args.input, step)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(
            prepared,
            scan=scan,
            examples=top_examples,
            thresholds=thresholds,
            max_mismatch=args.max_mismatch,
            tokenizer_name=args.tokenizer if tokenizer is not None else None,
        ),
        encoding="utf-8",
    )
    print_summary(output, scan, top_examples, thresholds)


def resolve_files(path: Path, *, step: int | None, all_steps: bool) -> tuple[list[Path], Path, int | None]:
    path = path.expanduser()
    if path.is_file():
        return [path], path.parent, step
    if not path.exists():
        raise FileNotFoundError(path)
    if all_steps:
        root = path
        resolved_step = None
    elif step is not None:
        root = path if path.name == f"step_{step}" else path / f"step_{step}"
        resolved_step = step
    else:
        current = step_number(path)
        if current is not None:
            root = path
            resolved_step = current
        else:
            step_dirs = [child for child in path.iterdir() if child.is_dir() and step_number(child) is not None]
            if not step_dirs:
                root = path
                resolved_step = None
            else:
                root = max(step_dirs, key=lambda child: step_number(child) or -1)
                resolved_step = step_number(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.jsonl"), key=path_sort_key)
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {root}")
    return files, root, resolved_step


def resolve_dppo_threshold(path: Path, key: str, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    config = find_trainer_config(path.expanduser())
    if config is not None and tomllib is not None:
        try:
            data = tomllib.loads(config.read_text(encoding="utf-8"))
            value = data.get("loss", {}).get(key)
            if value is not None:
                return float(value)
        except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
            pass
    return 0.2


def find_trainer_config(path: Path) -> Path | None:
    for base in [path, *path.parents]:
        candidate = base / "configs" / "trainer.toml"
        if candidate.is_file():
            return candidate
    return None


def step_number(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else None


def rank_number(path: Path) -> int | None:
    match = re.fullmatch(r"rank_(\d+)\.jsonl", path.name)
    return int(match.group(1)) if match else None


def path_sort_key(path: Path) -> tuple[int, int, str]:
    return (step_number(path.parent) or -1, rank_number(path) or -1, str(path))


def iter_records(
    files: list[Path],
    *,
    root: Path,
    rank: int | None,
    env_name: str | None,
):
    for file in files:
        with file.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = loads(line)
                if rank is not None and record.get("rank") != rank:
                    continue
                if env_name is not None and record.get("env_name") != env_name:
                    continue
                yield record, relative_source(file, root), line_number


def summarize_record(record: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    token_ids = record.get("token_ids") or []
    summary = empty_group()
    summary.update(
        {
            "step": record.get("step"),
            "rank": record.get("rank"),
            "env_name": record.get("env_name"),
            "micro_step": record.get("micro_step"),
            "micro_sequence_idx": record.get("micro_sequence_idx"),
            "export_sequence_idx": record.get("export_sequence_idx"),
            "training_mode": record.get("training_mode"),
            "token_count": len(token_ids),
            "trainable_tokens": 0,
            "max_index": None,
            "max_token_id": None,
            "max_disagree_index": None,
            "max_disagree_token_id": None,
        },
    )
    for index, token_id in enumerate(token_ids):
        token = compute_token(record, index, thresholds)
        if not token["loss_mask"]:
            continue
        summary["trainable_tokens"] += 1
        add_mask_counts(summary, token)
        mismatch = finite_float(token["mismatch_kl"])
        if mismatch is not None and (summary["max_mismatch"] is None or mismatch > summary["max_mismatch"]):
            summary["max_mismatch"] = mismatch
            summary["max_index"] = index
            summary["max_token_id"] = token_id
        if token["dppo_mask"] != token["icepop_mask"] and mismatch is not None:
            if summary["max_disagree_mismatch"] is None or mismatch > summary["max_disagree_mismatch"]:
                summary["max_disagree_mismatch"] = mismatch
                summary["max_disagree_index"] = index
                summary["max_disagree_token_id"] = token_id
    return summary


def empty_group() -> dict[str, Any]:
    return {
        "token_count": 0,
        "trainable_token_count": 0,
        "trainable_tokens": 0,
        "dppo_masked": 0,
        "icepop_masked": 0,
        "both_masked": 0,
        "neither_masked": 0,
        "dppo_only": 0,
        "icepop_only": 0,
        "disagree_count": 0,
        "dppo_high": 0,
        "dppo_low": 0,
        "icepop_high": 0,
        "icepop_low": 0,
        "max_mismatch": None,
        "max_disagree_mismatch": None,
    }


def add_mask_counts(target: dict[str, Any], token: dict[str, Any]) -> None:
    if token["dppo_mask"]:
        target["dppo_masked"] += 1
    if token["icepop_mask"]:
        target["icepop_masked"] += 1
    if token["dppo_mask"] and token["icepop_mask"]:
        target["both_masked"] += 1
    elif token["dppo_mask"]:
        target["dppo_only"] += 1
    elif token["icepop_mask"]:
        target["icepop_only"] += 1
    else:
        target["neither_masked"] += 1
    if token["dppo_mask"] != token["icepop_mask"]:
        target["disagree_count"] += 1
    if token["dppo_mask_kind"] == "high":
        target["dppo_high"] += 1
    elif token["dppo_mask_kind"] == "low":
        target["dppo_low"] += 1
    if token["icepop_mask_kind"] == "high":
        target["icepop_high"] += 1
    elif token["icepop_mask_kind"] == "low":
        target["icepop_low"] += 1


def merge_counts(target: dict[str, Any], summary: dict[str, Any]) -> None:
    for key in (
        "token_count",
        "dppo_masked",
        "icepop_masked",
        "both_masked",
        "neither_masked",
        "dppo_only",
        "icepop_only",
        "disagree_count",
        "dppo_high",
        "dppo_low",
        "icepop_high",
        "icepop_low",
    ):
        target[key] += int(summary.get(key) or 0)
    target["trainable_token_count"] += int(summary.get("trainable_tokens") or 0)
    target["max_mismatch"] = max_optional(target.get("max_mismatch"), summary.get("max_mismatch"))
    target["max_disagree_mismatch"] = max_optional(
        target.get("max_disagree_mismatch"),
        summary.get("max_disagree_mismatch"),
    )


def finalize_rates(group: dict[str, Any]) -> None:
    denom = int(group.get("trainable_token_count") or group.get("trainable_tokens") or 0)
    group["dppo_mask_rate"] = rate(group["dppo_masked"], denom)
    group["icepop_mask_rate"] = rate(group["icepop_masked"], denom)
    group["disagree_rate"] = rate(group["disagree_count"], denom)


def compute_token(record: dict[str, Any], index: int, thresholds: dict[str, float]) -> dict[str, Any]:
    loss_mask = bool(at(record.get("loss_mask"), index, False))
    advantage = finite_float(at(record.get("advantages"), index))
    prob_delta = finite_float(at(record.get("prob_delta"), index))
    ratio = finite_float(at(record.get("importance_ratio"), index))
    log_ratio = finite_float(at(record.get("log_importance_ratio"), index))
    if ratio is None and log_ratio is not None:
        ratio = exp_or_none(log_ratio)
    trainer_logprob = finite_float(at(record.get("trainer_logprobs"), index))
    inference_logprob = finite_float(at(record.get("inference_logprobs"), index))
    mismatch = finite_float(at(record.get("mismatch_kl"), index))

    dppo_kind = None
    if loss_mask and advantage is not None and prob_delta is not None:
        if advantage > 0 and prob_delta > thresholds["dppo_mask_high"]:
            dppo_kind = "high"
        elif advantage <= 0 and prob_delta < -thresholds["dppo_mask_low"]:
            dppo_kind = "low"

    icepop_kind = None
    if loss_mask and ratio is not None:
        if ratio < thresholds["icepop_ratio_low"]:
            icepop_kind = "low"
        elif ratio > thresholds["icepop_ratio_high"]:
            icepop_kind = "high"

    dppo_mask = dppo_kind is not None
    icepop_mask = icepop_kind is not None
    relation = "both" if dppo_mask and icepop_mask else "dppo_only" if dppo_mask else "icepop_only" if icepop_mask else "neither"
    return {
        "loss_mask": loss_mask,
        "advantage": advantage,
        "prob_delta": prob_delta,
        "importance_ratio": ratio,
        "log_importance_ratio": log_ratio,
        "trainer_logprob": trainer_logprob,
        "inference_logprob": inference_logprob,
        "trainer_prob": exp_or_none(trainer_logprob),
        "inference_prob": exp_or_none(inference_logprob),
        "mismatch_kl": mismatch,
        "dppo_mask": dppo_mask,
        "dppo_mask_kind": dppo_kind,
        "icepop_mask": icepop_mask,
        "icepop_mask_kind": icepop_kind,
        "mask_relation": relation,
    }


def collect_examples(
    examples: dict[str, list[tuple[tuple[float, int, int, int, int], dict[str, Any]]]],
    record: dict[str, Any],
    summary: dict[str, Any],
    thresholds: dict[str, float],
    *,
    limit: int = 20,
) -> None:
    for index, token_id in enumerate(record.get("token_ids") or []):
        token = compute_token(record, index, thresholds)
        if not token["loss_mask"]:
            continue
        mismatch = finite_float(token["mismatch_kl"]) or 0.0
        row = {
            "step": record.get("step"),
            "rank": record.get("rank"),
            "env_name": record.get("env_name"),
            "source_file": summary.get("source_file"),
            "source_line": summary.get("source_line"),
            "index": index,
            "id": token_id,
            "text": None,
            **token,
        }
        key = token["mask_relation"]
        if key in examples:
            push_example(examples[key], mismatch, row, limit)
        if key == "neither" and mismatch >= 3.0:
            push_example(examples["unmasked_high_mismatch"], mismatch, row, limit)


def push_example(
    bucket: list[tuple[tuple[float, int, int, int, int], dict[str, Any]]],
    score: float,
    row: dict[str, Any],
    limit: int,
) -> None:
    key = (
        score,
        int(row.get("step") or -1),
        int(row.get("rank") or -1),
        int(row.get("source_line") or -1),
        int(row["index"]),
    )
    if len(bucket) < limit:
        heapq.heappush(bucket, (key, row))
    elif key > bucket[0][0]:
        heapq.heapreplace(bucket, (key, row))


def prepare_record(
    record: dict[str, Any],
    *,
    summary: dict[str, Any],
    ordinal: int,
    thresholds: dict[str, float],
    tokenizer: Any | None,
    window_tokens: int,
    full_record: bool,
) -> dict[str, Any]:
    token_ids = record.get("token_ids") or []
    focus = summary.get("max_disagree_index")
    if focus is None:
        focus = summary.get("max_index")
    if focus is None:
        focus = len(token_ids) // 2
    if full_record:
        start, end = 0, len(token_ids)
    else:
        half = window_tokens // 2
        start = max(0, int(focus) - half)
        end = min(len(token_ids), start + window_tokens)
        start = max(0, end - window_tokens)

    texts = decode_tokens(tokenizer, token_ids[start:end])
    tokens = []
    for offset, token_id in enumerate(token_ids[start:end], start=start):
        token = compute_token(record, offset, thresholds)
        token.update(
            {
                "id": token_id,
                "text": texts[offset - start],
                "index": offset,
                "position": at(record.get("position_ids"), offset),
                "reward": at(record.get("rewards"), offset),
                "entropy": finite_float(at(record.get("entropy"), offset)),
            },
        )
        tokens.append(token)

    meta = {
        **summary,
        "ordinal": ordinal,
        "window_start": start,
        "window_end": end,
    }
    meta["label"] = record_label(meta)
    meta["subtitle"] = record_subtitle(meta)
    meta["search_text"] = f"{meta['label']} {meta['subtitle']}".lower()
    return {"meta": meta, "tokens": tokens}


def decode_tokens(tokenizer: Any | None, token_ids: list[int]) -> list[str]:
    if tokenizer is None:
        return [f"[{token_id}]" for token_id in token_ids]
    decoded = []
    for token_id in token_ids:
        try:
            text = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        decoded.append(text)
    return decoded


def decode_examples(examples: dict[str, list[dict[str, Any]]], tokenizer: Any | None) -> None:
    if tokenizer is None:
        return
    for rows in examples.values():
        for row in rows:
            row["text"] = decode_tokens(tokenizer, [row["id"]])[0]


def render_html(
    records: list[dict[str, Any]],
    *,
    scan: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    max_mismatch: float | None,
    tokenizer_name: str | None,
) -> str:
    scale = mismatch_scale([token for record in records for token in record["tokens"]], max_mismatch)
    payload = {
        "records": records,
        "scan": scan,
        "examples": examples,
        "thresholds": thresholds,
        "scale": scale,
        "tokenizer": tokenizer_name,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token Mask Visualizer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #111827;
      --muted: #667085;
      --dppo: #2563eb;
      --icepop: #d97706;
      --both: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
    .app {{ display: grid; grid-template-columns: 340px minmax(0, 1fr) 360px; min-height: 100vh; }}
    aside, main, .right {{ padding: 16px; }}
    aside, .right {{ background: var(--panel); border-color: var(--line); overflow: auto; max-height: 100vh; }}
    aside {{ border-right: 1px solid var(--line); }}
    .right {{ border-left: 1px solid var(--line); }}
    h1, h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    h2 {{ margin-top: 20px; font-size: 13px; text-transform: uppercase; color: var(--muted); }}
    .scan, .hint {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .controls {{ display: grid; gap: 10px; margin-top: 14px; }}
    label {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 13px; color: #344054; }}
    input[type="number"], input[type="search"] {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; background: #fff; color: var(--text); }}
    input[type="number"] {{ max-width: 96px; }}
    button {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--text); padding: 8px 10px; cursor: pointer; text-align: left; }}
    button:hover {{ border-color: #98a2b3; }}
    button.active {{ border-color: #111827; background: #eef2ff; }}
    .record-list {{ display: grid; gap: 8px; margin-top: 10px; }}
    .record-title {{ font-weight: 650; font-size: 13px; }}
    .record-subtitle {{ color: var(--muted); font-size: 12px; line-height: 1.35; margin-top: 3px; }}
    main {{ overflow: auto; max-height: 100vh; }}
    .topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
    .chips {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .chip {{ display: inline-flex; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); background: #fff; font-size: 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 9px; font-size: 12px; color: var(--muted); }}
    .metric strong {{ display: block; color: var(--text); font-size: 16px; margin-bottom: 2px; }}
    .tokens {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; line-height: 1.9; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; white-space: pre-wrap; }}
    .token {{ border-radius: 3px; padding: 2px 1px; margin: 0 1px; cursor: default; }}
    .legend {{ display: grid; gap: 6px; margin-top: 12px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
    .red {{ background: rgba(220, 38, 38, 0.72); }}
    .blue {{ background: var(--dppo); }}
    .orange {{ background: var(--icepop); }}
    .purple {{ background: var(--both); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 4px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; width: 130px; }}
    .details {{ background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 120px; overflow-wrap: anywhere; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 300px minmax(0, 1fr); }}
      .right {{ grid-column: 1 / -1; max-height: none; border-left: 0; border-top: 1px solid var(--line); }}
      aside, main {{ max-height: none; }}
    }}
    @media (max-width: 760px) {{
      .app {{ display: block; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Mask Comparison</h1>
      <div id="scan" class="scan"></div>
      <div class="controls">
        <label><span>KL gradient</span><input id="layer-kl" type="checkbox" checked></label>
        <label><span>DPPO overlay</span><input id="layer-dppo" type="checkbox" checked></label>
        <label><span>IcePop overlay</span><input id="layer-icepop" type="checkbox" checked></label>
        <label><span>DPPO low</span><input id="dppo-low" type="number" min="0" step="0.01"></label>
        <label><span>DPPO high</span><input id="dppo-high" type="number" min="0" step="0.01"></label>
        <label><span>IcePop low</span><input id="icepop-low" type="number" min="0" step="0.01"></label>
        <label><span>IcePop high</span><input id="icepop-high" type="number" min="0" step="0.1"></label>
        <button id="reset-thresholds" type="button">Reset thresholds</button>
      </div>
      <div class="legend">
        <div><span class="swatch red"></span>red intensity = mismatch KL</div>
        <div><span class="swatch blue"></span>blue bottom rule = DPPO masked</div>
        <div><span class="swatch orange"></span>orange top rule = IcePop masked</div>
        <div><span class="swatch purple"></span>purple fill = both masked when KL is hidden</div>
      </div>
      <h2>Records</h2>
      <input id="record-search" type="search" placeholder="Filter records">
      <div id="record-list" class="record-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1 id="record-title"></h1>
          <div id="record-chips" class="chips"></div>
        </div>
      </div>
      <div id="metrics" class="metrics"></div>
      <div id="tokens" class="tokens"></div>
    </main>
    <section class="right">
      <h1>Token Details</h1>
      <div id="details" class="details muted">Hover a token.</div>
      <h2>Top Tokens</h2>
      <div id="examples"></div>
    </section>
  </div>
  <script id="payload" type="application/json">{json_script(payload)}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const records = payload.records;
    const scan = payload.scan;
    const defaultThresholds = {{...payload.thresholds}};
    let thresholds = {{...payload.thresholds}};
    let activeIndex = 0;
    const layers = {{kl: true, dppo: true, icepop: true}};

    const el = {{
      scan: document.getElementById("scan"),
      metrics: document.getElementById("metrics"),
      tokens: document.getElementById("tokens"),
      title: document.getElementById("record-title"),
      chips: document.getElementById("record-chips"),
      details: document.getElementById("details"),
      examples: document.getElementById("examples"),
      recordList: document.getElementById("record-list"),
      recordSearch: document.getElementById("record-search"),
    }};
    const thresholdInputs = {{
      dppo_mask_low: document.getElementById("dppo-low"),
      dppo_mask_high: document.getElementById("dppo-high"),
      icepop_ratio_low: document.getElementById("icepop-low"),
      icepop_ratio_high: document.getElementById("icepop-high"),
    }};

    el.scan.textContent = `${{scan.records_matched}} records, ${{scan.trainable_token_count.toLocaleString()}} trainable tokens, tokenizer ${{payload.tokenizer || "none"}}`;
    for (const [key, input] of Object.entries(thresholdInputs)) input.value = thresholds[key];
    document.getElementById("layer-kl").addEventListener("change", (event) => {{ layers.kl = event.target.checked; repaint(); }});
    document.getElementById("layer-dppo").addEventListener("change", (event) => {{ layers.dppo = event.target.checked; repaint(); }});
    document.getElementById("layer-icepop").addEventListener("change", (event) => {{ layers.icepop = event.target.checked; repaint(); }});
    for (const [key, input] of Object.entries(thresholdInputs)) {{
      input.addEventListener("input", () => {{
        const value = Number(input.value);
        if (Number.isFinite(value) && value >= 0) {{
          thresholds[key] = value;
          renderRecord(activeIndex);
        }}
      }});
    }}
    document.getElementById("reset-thresholds").addEventListener("click", () => {{
      thresholds = {{...defaultThresholds}};
      for (const [key, input] of Object.entries(thresholdInputs)) input.value = thresholds[key];
      renderRecord(activeIndex);
    }});
    el.recordSearch.addEventListener("input", renderNav);

    function renderNav() {{
      const query = el.recordSearch.value.trim().toLowerCase();
      el.recordList.replaceChildren();
      records.forEach((record, index) => {{
        if (query && !record.meta.search_text.includes(query)) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = index === activeIndex ? "active" : "";
        button.addEventListener("click", () => renderRecord(index));
        button.innerHTML = `<div class="record-title">${{escapeHtml(record.meta.label)}}</div><div class="record-subtitle">${{escapeHtml(record.meta.subtitle)}}</div>`;
        el.recordList.appendChild(button);
      }});
    }}

    function renderRecord(index) {{
      activeIndex = index;
      const record = records[index];
      el.title.textContent = record.meta.label;
      el.chips.replaceChildren();
      [
        `disagree ${{record.meta.disagree_count}}`,
        `DPPO ${{record.meta.dppo_masked}}`,
        `IcePop ${{record.meta.icepop_masked}}`,
        `max KL ${{fmt(record.meta.max_mismatch)}}`,
        `max diff KL ${{fmt(record.meta.max_disagree_mismatch)}}`,
        `window ${{record.meta.window_start}}-${{record.meta.window_end}}`,
        `${{record.meta.source_file}}:${{record.meta.source_line}}`,
      ].forEach((text) => {{
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = text;
        el.chips.appendChild(chip);
      }});
      el.tokens.replaceChildren();
      for (const rawToken of record.tokens) el.tokens.appendChild(renderToken(rawToken));
      renderMetrics(record);
      renderNav();
    }}

    function renderToken(rawToken) {{
      const token = runtimeToken(rawToken);
      const span = document.createElement("span");
      span.className = "token";
      span.textContent = rawToken.text === "" || rawToken.text === null ? `[${{rawToken.id}}]` : rawToken.text;
      span.__rawToken = rawToken;
      span.title = tokenTitle(token);
      applyTokenStyle(span, token);
      span.addEventListener("mouseenter", () => renderDetails(runtimeToken(rawToken)));
      return span;
    }}

    function runtimeToken(rawToken) {{
      if (!rawToken.loss_mask) {{
        return {{...rawToken, dppo_mask: false, dppo_mask_kind: null, icepop_mask: false, icepop_mask_kind: null, mask_relation: "neither"}};
      }}
      const advantage = finite(rawToken.advantage);
      const probDelta = finite(rawToken.prob_delta);
      const ratio = finite(rawToken.importance_ratio);
      let dppoKind = null;
      if (advantage !== null && probDelta !== null) {{
        if (advantage > 0 && probDelta > thresholds.dppo_mask_high) dppoKind = "high";
        else if (advantage <= 0 && probDelta < -thresholds.dppo_mask_low) dppoKind = "low";
      }}
      let icepopKind = null;
      if (ratio !== null) {{
        if (ratio < thresholds.icepop_ratio_low) icepopKind = "low";
        else if (ratio > thresholds.icepop_ratio_high) icepopKind = "high";
      }}
      const token = {{
        ...rawToken,
        dppo_mask: dppoKind !== null,
        dppo_mask_kind: dppoKind,
        icepop_mask: icepopKind !== null,
        icepop_mask_kind: icepopKind,
      }};
      token.mask_relation = token.dppo_mask && token.icepop_mask ? "both" : token.dppo_mask ? "dppo_only" : token.icepop_mask ? "icepop_only" : "neither";
      token.live_dppo_rule = advantage > 0 ? `prob_delta > ${{thresholds.dppo_mask_high}}` : `prob_delta < -${{thresholds.dppo_mask_low}}`;
      token.live_icepop_rule = `ratio < ${{thresholds.icepop_ratio_low}} or > ${{thresholds.icepop_ratio_high}}`;
      return token;
    }}

    function applyTokenStyle(span, token) {{
      span.style.backgroundColor = "transparent";
      span.style.color = token.loss_mask ? "#111827" : "#667085";
      span.style.borderBottom = "0";
      span.style.borderTop = "0";
      if (!token.loss_mask) return;
      const mismatch = finite(token.mismatch_kl);
      const alpha = mismatch === null ? 0 : Math.max(0, Math.min(1, mismatch / payload.scale));
      if (layers.kl && alpha > 0) {{
        span.style.backgroundColor = `rgba(220, 38, 38, ${{(0.08 + 0.72 * alpha).toFixed(3)}})`;
        if (alpha > 0.8) span.style.color = "#ffffff";
      }} else if (!layers.kl) {{
        if (layers.dppo && layers.icepop && token.mask_relation === "both") span.style.backgroundColor = "rgba(124, 58, 237, 0.55)";
        else if (layers.dppo && token.dppo_mask) span.style.backgroundColor = "rgba(37, 99, 235, 0.35)";
        else if (layers.icepop && token.icepop_mask) span.style.backgroundColor = "rgba(217, 119, 6, 0.35)";
      }}
      if (layers.dppo && token.dppo_mask) span.style.borderBottom = "2px solid #2563eb";
      if (layers.icepop && token.icepop_mask) span.style.borderTop = "2px solid #d97706";
    }}

    function renderMetrics(record) {{
      const stats = runtimeStats(record);
      const rows = [
        ["Global DPPO", `${{scan.dppo_masked.toLocaleString()}} (${{pct(scan.dppo_mask_rate)}})`],
        ["Global IcePop", `${{scan.icepop_masked.toLocaleString()}} (${{pct(scan.icepop_mask_rate)}})`],
        ["Global disagree", `${{scan.disagree_count.toLocaleString()}} (${{pct(scan.disagree_rate)}})`],
        ["Window DPPO live", `${{stats.dppo}}/${{stats.trainable}} (${{pct(stats.dppo / Math.max(stats.trainable, 1))}})`],
        ["Window IcePop live", `${{stats.icepop}}/${{stats.trainable}} (${{pct(stats.icepop / Math.max(stats.trainable, 1))}})`],
        ["Window disagree live", `${{stats.disagree}}/${{stats.trainable}} (${{pct(stats.disagree / Math.max(stats.trainable, 1))}})`],
      ];
      el.metrics.replaceChildren();
      for (const [name, value] of rows) {{
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<strong>${{escapeHtml(value)}}</strong>${{escapeHtml(name)}}`;
        el.metrics.appendChild(div);
      }}
    }}

    function runtimeStats(record) {{
      const stats = {{trainable: 0, dppo: 0, icepop: 0, disagree: 0}};
      for (const rawToken of record.tokens) {{
        const token = runtimeToken(rawToken);
        if (!token.loss_mask) continue;
        stats.trainable += 1;
        if (token.dppo_mask) stats.dppo += 1;
        if (token.icepop_mask) stats.icepop += 1;
        if (token.dppo_mask !== token.icepop_mask) stats.disagree += 1;
      }}
      return stats;
    }}

    function repaint() {{
      document.querySelectorAll(".token").forEach((span) => {{
        const token = runtimeToken(span.__rawToken);
        applyTokenStyle(span, token);
        span.title = tokenTitle(token);
      }});
      renderMetrics(records[activeIndex]);
    }}

    function renderDetails(token) {{
      const preferred = ["text", "id", "index", "loss_mask", "mask_relation", "dppo_mask", "dppo_mask_kind", "icepop_mask", "icepop_mask_kind", "mismatch_kl", "importance_ratio", "prob_delta", "advantage", "inference_prob", "trainer_prob", "inference_logprob", "trainer_logprob", "live_dppo_rule", "live_icepop_rule"];
      const table = document.createElement("table");
      const body = document.createElement("tbody");
      const seen = new Set();
      for (const key of preferred) {{
        if (Object.prototype.hasOwnProperty.call(token, key)) {{
          addDetail(body, key, token[key]);
          seen.add(key);
        }}
      }}
      for (const key of Object.keys(token).sort()) if (!seen.has(key)) addDetail(body, key, token[key]);
      table.appendChild(body);
      el.details.className = "details";
      el.details.replaceChildren(table);
    }}

    function addDetail(body, key, value) {{
      const row = document.createElement("tr");
      const th = document.createElement("th");
      const td = document.createElement("td");
      th.textContent = key;
      td.textContent = fmt(value);
      row.appendChild(th);
      row.appendChild(td);
      body.appendChild(row);
    }}

    function renderExamples() {{
      const table = document.createElement("table");
      const head = document.createElement("thead");
      head.innerHTML = "<tr><th>type</th><th>token</th><th>KL</th><th>ratio</th><th>prob_delta</th><th>where</th></tr>";
      const body = document.createElement("tbody");
      for (const [name, rows] of Object.entries(payload.examples)) {{
        for (const row of rows.slice(0, 6)) {{
          const tr = document.createElement("tr");
          const token = row.text === null || row.text === undefined ? `[${{row.id}}]` : row.text;
          tr.innerHTML = `<td>${{escapeHtml(name)}}</td><td>${{escapeHtml(token)}}</td><td>${{fmt(row.mismatch_kl)}}</td><td>${{fmt(row.importance_ratio)}}</td><td>${{fmt(row.prob_delta)}}</td><td>s${{row.step}} r${{row.rank}} i${{row.index}}</td>`;
          body.appendChild(tr);
        }}
      }}
      table.appendChild(head);
      table.appendChild(body);
      el.examples.replaceChildren(table);
    }}

    function tokenTitle(token) {{
      return `DPPO ${{token.dppo_mask}} (${{token.dppo_mask_kind}})\nIcePop ${{token.icepop_mask}} (${{token.icepop_mask_kind}})\nKL ${{fmt(token.mismatch_kl)}}\nratio ${{fmt(token.importance_ratio)}}\nprob_delta ${{fmt(token.prob_delta)}}`;
    }}
    function finite(value) {{
      if (value === null || value === undefined) return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }}
    function fmt(value) {{
      if (value === null || value === undefined) return "null";
      if (typeof value === "number") {{
        if (!Number.isFinite(value)) return String(value);
        if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) return value.toExponential(6);
        return Number(value.toPrecision(7)).toString();
      }}
      return String(value);
    }}
    function pct(value) {{ return `${{(100 * value).toFixed(3)}}%`; }}
    function escapeHtml(value) {{
      return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }}
    renderExamples();
    renderRecord(0);
  </script>
</body>
</html>
"""


def record_label(meta: dict[str, Any]) -> str:
    parts = [f"#{meta['ordinal']}"]
    for key, label in (("step", "step"), ("rank", "rank"), ("env_name", "env"), ("export_sequence_idx", "seq")):
        if meta.get(key) is not None:
            parts.append(f"{label} {meta[key]}")
    return " | ".join(parts)


def record_subtitle(meta: dict[str, Any]) -> str:
    return (
        f"disagree {meta.get('disagree_count')} | DPPO {meta.get('dppo_masked')} | "
        f"IcePop {meta.get('icepop_masked')} | max diff KL {format_number(meta.get('max_disagree_mismatch'))} | "
        f"{meta.get('source_file')}:{meta.get('source_line')}"
    )


def mismatch_scale(tokens: list[dict[str, Any]], override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--max-mismatch must be positive.")
        return override
    values = [finite_float(token.get("mismatch_kl")) for token in tokens if token.get("loss_mask")]
    values = sorted(value for value in values if value is not None and value > 0)
    if not values:
        return 1.0
    index = min(max(round((len(values) - 1) * 0.95), 0), len(values) - 1)
    return max(values[index], 0.05)


def load_tokenizer(name: str, *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=local_files_only, trust_remote_code=True)


def loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def relative_source(file: Path, root: Path) -> str:
    try:
        return str(file.relative_to(root))
    except ValueError:
        return str(file)


def at(values: Any, index: int | None, default: Any = None) -> Any:
    if values is None or index is None:
        return default
    return values[index] if 0 <= index < len(values) else default


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def exp_or_none(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    try:
        return math.exp(number)
    except OverflowError:
        return None


def max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_script(value: Any) -> str:
    payload = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def default_output(path: Path, step: int | None) -> Path:
    if path.is_file():
        return path.with_name(path.stem + "_mask_visualizer.html")
    if step is not None:
        return path / f"token_mask_visualizer_step_{step}.html"
    return path / "token_mask_visualizer.html"


def format_number(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "null"
    if abs(number) >= 1000 or (abs(number) > 0 and abs(number) < 0.0001):
        return f"{number:.6e}"
    return f"{number:.6g}"


def print_summary(
    output: Path,
    scan: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
) -> None:
    print(output)
    print(
        "thresholds "
        f"DPPO prob_delta=[-{thresholds['dppo_mask_low']}, +{thresholds['dppo_mask_high']}], "
        f"IcePop ratio=[{thresholds['icepop_ratio_low']}, {thresholds['icepop_ratio_high']}]",
    )
    print(
        f"records={scan['records_matched']} trainable={scan['trainable_token_count']} "
        f"dppo_masked={scan['dppo_masked']} ({100 * scan['dppo_mask_rate']:.3f}%) "
        f"icepop_masked={scan['icepop_masked']} ({100 * scan['icepop_mask_rate']:.3f}%) "
        f"disagree={scan['disagree_count']} ({100 * scan['disagree_rate']:.3f}%)",
    )
    for key in ("icepop_only", "dppo_only", "both", "unmasked_high_mismatch"):
        rows = examples.get(key) or []
        if not rows:
            continue
        row = rows[0]
        print(
            f"top_{key}: token={row.get('text')!r} id={row['id']} step={row['step']} "
            f"rank={row['rank']} idx={row['index']} kl={format_number(row['mismatch_kl'])} "
            f"ratio={format_number(row['importance_ratio'])} prob_delta={format_number(row['prob_delta'])} "
            f"adv={format_number(row['advantage'])}",
        )


if __name__ == "__main__":
    main()
