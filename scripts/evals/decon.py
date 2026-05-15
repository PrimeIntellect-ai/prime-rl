"""N-gram decontamination check for dolci-debate-sft-v1 vs eval prompts.

Strategy: word-level 13-grams (GPT-3 convention). Build a reverse index
`ngram_hash → {(eval, row_idx)}` from the evals (small), then stream the
training dataset and tag each eval row with the number of its 13-grams that
show up verbatim in training.

Report per-eval:
  - loose contamination: % of eval rows with ≥1 13-gram match (GPT-3 threshold)
  - strict contamination: % of eval rows where ≥50% of 13-grams match
  - leaderboard of worst-leaking rows

Normalization: lowercase + strip punctuation + whitespace-split. Matches the
canonical dolma / olmes decon pipeline closely enough for sweep pre-flight.

Usage:
    uv run python -m scripts.evals.decon --output-dir outputs/evals/decon
    uv run python -m scripts.evals.decon --subsets wildchat dolci-precise-if  # debug
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from ._server import resolve_path_args

TRAIN_REPO = "joanvelja/dolci-debate-sft-v1"
TRAIN_SUBSETS = [
    "wildchat",
    "tulu-3-persona-math",
    "dolci-precise-if",
    "dolci-openthoughts-sci",
    "evol-codealpaca",
    "dolci-python-algo",
    "flan",
    "tulu-3-persona-algebra",
    "openmathinstruct-2",
    "openassistant",
    "tablegpt",
    "sciriff",
]

NGRAM_SIZE = 13
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase + alphanum word-tokenize. Matches GPT-3 / dolma decon normalization."""
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int = NGRAM_SIZE) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


# -------------------- eval prompt extraction --------------------


def _load_ifeval_prompts() -> list[str]:
    ds = load_dataset("HuggingFaceH4/ifeval", split="train")
    return [r["prompt"] for r in ds]


def _load_gsm8k_prompts() -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [r["question"] for r in ds]


def _load_mmlu_prompts() -> list[str]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Question alone (choices come from fixed templates we don't want matching).
    return [r["question"] for r in ds]


def _load_mtbench_prompts() -> list[str]:
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    out = []
    for r in ds:
        turns = r.get("prompt") or r.get("turns") or []
        if isinstance(turns, str):
            out.append(turns)
        else:
            out.extend(t for t in turns if isinstance(t, str))
    return out


EVALS: dict[str, callable] = {
    "ifeval": _load_ifeval_prompts,
    "gsm8k": _load_gsm8k_prompts,
    "mmlu": _load_mmlu_prompts,
    "mtbench": _load_mtbench_prompts,
}


# -------------------- training-row text extraction --------------------


def _extract_training_text(row: dict) -> str:
    """Dolci rows vary: chat format (`messages`), raw strings, nested dicts."""
    if "messages" in row and isinstance(row["messages"], list):
        parts = []
        for m in row["messages"]:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)
    for key in ("text", "prompt", "question", "input", "instruction"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    raise ValueError(f"Unknown training row schema: keys={sorted(row.keys())[:10]}")


# -------------------- decon pipeline --------------------


def build_eval_index(
    evals: dict[str, list[str]],
) -> tuple[
    dict[tuple[str, ...], set[tuple[str, int]]],
    dict[tuple[str, int], int],
]:
    """Reverse index: each 13-gram → set of (eval_name, row_idx) it appears in.

    Also returns per-row total 13-gram count (for strict-fraction calc).
    """
    index: dict[tuple[str, ...], set[tuple[str, int]]] = defaultdict(set)
    counts: dict[tuple[str, int], int] = {}
    for eval_name, prompts in evals.items():
        for idx, prompt in enumerate(prompts):
            toks = _tokens(prompt)
            row_ngrams = list(_ngrams(toks))
            counts[(eval_name, idx)] = len(row_ngrams)
            for ng in set(row_ngrams):
                index[ng].add((eval_name, idx))
    return index, counts


def run(
    output_dir: Path,
    *,
    subsets: list[str] | None = None,
    sample_cap: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = subsets or TRAIN_SUBSETS

    print("[decon] loading eval prompts...", flush=True)
    t0 = time.time()
    evals = {name: loader() for name, loader in EVALS.items()}
    for name, prompts in evals.items():
        print(f"  {name}: {len(prompts)} prompts", flush=True)

    print("\n[decon] building 13-gram index from evals...", flush=True)
    index, counts = build_eval_index(evals)
    total_unique_ngrams = len(index)
    print(f"  {total_unique_ngrams} unique 13-grams across all evals (from {sum(counts.values())} total)", flush=True)

    # hits[(eval_name, row_idx)] = set of 13-grams from that row that were seen in training
    hits: dict[tuple[str, int], set[tuple[str, ...]]] = defaultdict(set)

    print(f"\n[decon] streaming {len(subsets)} training subsets...", flush=True)
    for subset in subsets:
        t_sub = time.time()
        n_rows = 0
        n_hits = 0
        try:
            ds = load_dataset(TRAIN_REPO, subset, split="train", streaming=True)
        except Exception as exc:
            print(f"  {subset}: LOAD FAILED — {exc}", flush=True)
            continue
        for row in ds:
            n_rows += 1
            if sample_cap and n_rows > sample_cap:
                break
            text = _extract_training_text(row)
            toks = _tokens(text)
            row_hits = 0
            seen: set[tuple[str, ...]] = set()
            for ng in _ngrams(toks):
                if ng in seen:
                    continue
                seen.add(ng)
                matches = index.get(ng)
                if matches:
                    row_hits += 1
                    for key in matches:
                        hits[key].add(ng)
            n_hits += row_hits
            if n_rows % 50000 == 0:
                elapsed = time.time() - t_sub
                print(f"    {subset}: {n_rows} rows, {n_hits} ngram hits so far ({elapsed:.0f}s)", flush=True)
        elapsed = time.time() - t_sub
        print(f"  {subset}: {n_rows} rows scanned in {elapsed:.0f}s, {n_hits} ngram-hits", flush=True)

    # Aggregate per-eval
    report: dict = {
        "elapsed_s": time.time() - t0,
        "ngram_size": NGRAM_SIZE,
        "subsets_scanned": subsets,
        "sample_cap": sample_cap,
        "per_eval": {},
    }
    for eval_name, prompts in evals.items():
        n_rows = len(prompts)
        loose_hits = 0
        strict_hits = 0
        worst: list[tuple[int, float, int, int]] = []  # (idx, frac_matched, n_matched, total)
        for idx in range(n_rows):
            total = counts.get((eval_name, idx), 0)
            if total == 0:
                continue
            matched = len(hits.get((eval_name, idx), set()))
            if matched >= 1:
                loose_hits += 1
            frac = matched / total
            if frac >= 0.5:
                strict_hits += 1
            if matched > 0:
                worst.append((idx, frac, matched, total))
        worst.sort(key=lambda t: (-t[1], -t[2]))
        worst_top = [
            {"idx": i, "frac_matched": round(f, 4), "matched": m, "total": t, "prompt_preview": prompts[i][:200]}
            for (i, f, m, t) in worst[:20]
        ]
        report["per_eval"][eval_name] = {
            "n_rows": n_rows,
            "loose_contamination_rate": loose_hits / n_rows if n_rows else 0.0,
            "strict_contamination_rate": strict_hits / n_rows if n_rows else 0.0,
            "worst_20": worst_top,
        }

    report_path = output_dir / "decon_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(f"{'eval':<12} {'N':>8} {'loose (≥1)':>14} {'strict (≥50%)':>16}")
    print("-" * 70)
    for eval_name, r in report["per_eval"].items():
        loose_pct = 100.0 * r["loose_contamination_rate"]
        strict_pct = 100.0 * r["strict_contamination_rate"]
        print(f"  {eval_name:<10} {r['n_rows']:>8} {loose_pct:>13.2f}% {strict_pct:>15.3f}%")
    print("=" * 70)
    print(f"\n[decon] wrote {report_path}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/evals/decon"))
    p.add_argument("--subsets", nargs="+", default=None, help="Override default subset list (debug).")
    p.add_argument("--sample-cap", type=int, default=None, help="Cap rows per subset (debug).")
    return p.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "output_dir")
    run(args.output_dir, subsets=args.subsets, sample_cap=args.sample_cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
