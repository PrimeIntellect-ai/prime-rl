"""Generate a fingerprint-based filter for dolci-precise-if ↔ IFEval contamination.

Scans one training subset (default `dolci-precise-if`), computes per-row the
maximum 13-gram coverage against any IFEval prompt, and flags rows where
`max_coverage ≥ threshold`. Each flagged row is identified by a SHA-256 of
its canonical `messages` serialization so the filter survives shuffles.

Output:
    <output_dir>/filter.json
        {
          "threshold": 0.5,
          "subset": "dolci-precise-if",
          "n_rows": 136820,
          "n_contaminated": ...,
          "contamination_rate": ...,
          "fingerprints": ["<sha256>", ...],
        }
    <output_dir>/details.json  (per-row trace for audit)

Usage:
    uv run python -m scripts.evals.decon_filter --threshold 0.5
    uv run python -m scripts.evals.decon_filter --threshold 0.3 --subset dolci-precise-if
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from ._server import resolve_path_args
from .decon import (
    NGRAM_SIZE,
    TRAIN_REPO,
    _extract_training_text,
    _ngrams,
    _tokens,
)

IFEVAL_DATASET = "HuggingFaceH4/ifeval"
IFEVAL_SPLIT = "train"
DEFAULT_SUBSET = "dolci-precise-if"


def _canonical_messages(row: dict) -> str:
    """Canonical JSON serialization of chat-format row for hashing."""
    msgs = row.get("messages")
    if isinstance(msgs, list):
        return json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)


def _fingerprint(row: dict) -> str:
    return hashlib.sha256(_canonical_messages(row).encode("utf-8")).hexdigest()


def run(
    output_dir: Path,
    *,
    threshold: float = 0.5,
    subset: str = DEFAULT_SUBSET,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[decon_filter] threshold={threshold}  subset={subset}", flush=True)
    print("[decon_filter] loading IFEval prompts and building per-prompt 13-gram sets...", flush=True)
    t0 = time.time()
    ds_if = load_dataset(IFEVAL_DATASET, split=IFEVAL_SPLIT)
    ifeval_grams: list[set] = []
    for r in ds_if:
        ifeval_grams.append(set(_ngrams(_tokens(r["prompt"]))))
    print(
        f"  {len(ifeval_grams)} IFEval prompts, median grams/prompt = {sorted(len(g) for g in ifeval_grams)[len(ifeval_grams) // 2]}",
        flush=True,
    )

    # Reverse index: ngram → {ifeval_row_idx, ...}
    rev: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for i, s in enumerate(ifeval_grams):
        for ng in s:
            rev[ng].add(i)
    print(f"  {len(rev)} unique IFEval 13-grams", flush=True)

    print(f"\n[decon_filter] scanning {TRAIN_REPO}/{subset}...", flush=True)
    ds = load_dataset(TRAIN_REPO, subset, split="train", streaming=True)

    contaminated: list[dict] = []
    total = 0
    t_scan = time.time()
    for row in ds:
        total += 1
        text = _extract_training_text(row)
        tr_grams = set(_ngrams(_tokens(text)))
        coverage: dict[int, int] = defaultdict(int)
        for ng in tr_grams:
            for i in rev.get(ng, ()):
                coverage[i] += 1
        max_frac = 0.0
        worst = -1
        for i, cov in coverage.items():
            total_if = len(ifeval_grams[i])
            if total_if == 0:
                continue
            frac = cov / total_if
            if frac > max_frac:
                max_frac = frac
                worst = i
        if max_frac >= threshold:
            contaminated.append(
                {
                    "fp": _fingerprint(row),
                    "max_frac": round(max_frac, 4),
                    "worst_ifeval_idx": worst,
                    "tr_preview": text[:200],
                }
            )
        if total % 20000 == 0:
            rate = 100.0 * len(contaminated) / total
            print(
                f"    {total} rows scanned, {len(contaminated)} contaminated ({rate:.2f}%) — {time.time() - t_scan:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    result = {
        "threshold": threshold,
        "ngram_size": NGRAM_SIZE,
        "subset": subset,
        "n_rows": total,
        "n_contaminated": len(contaminated),
        "contamination_rate": len(contaminated) / total if total else 0.0,
        "elapsed_s": round(elapsed, 1),
        "fingerprints": [c["fp"] for c in contaminated],
    }
    (output_dir / "filter.json").write_text(json.dumps(result, indent=2))
    (output_dir / "details.json").write_text(json.dumps(contaminated, indent=2))

    print(
        f"\n[decon_filter] {len(contaminated)}/{total} rows flagged "
        f"({100 * len(contaminated) / total:.2f}%) at threshold {threshold}",
        flush=True,
    )
    print(f"[decon_filter] wrote {output_dir}/filter.json ({len(result['fingerprints'])} fingerprints)", flush=True)
    print(f"[decon_filter] wrote {output_dir}/details.json (audit trail)", flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/decon/ifeval_filter"))
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Coverage threshold above which a row is flagged (default 0.5)."
    )
    p.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
    return p.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "output_dir")
    run(args.output_dir, threshold=args.threshold, subset=args.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
