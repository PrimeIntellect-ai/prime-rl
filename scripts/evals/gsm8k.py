"""Native GSM8K — paired base vs ckpt, zero-shot CoT.

Dataset: `openai/gsm8k` config `main` split `test` (1319 rows). Each row has
`question` and `answer` ending with `#### <number>`.

Scoring: flexible-extract — last number anywhere in the response wins.
We skip the strict `#### N` variant; SFT responses rarely emit it verbatim.

Output: `<output_dir>/gsm8k.json` with base/ckpt/delta accuracy + per-row
details.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ._server import AccResult, PhaseHandle, complete_batch, paired_eval, resolve_path_args

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"

# Google's canonical GSM8K answer delimiter.
_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
# Flexible extract: last number anywhere in the response.
_NUMBER_RE = re.compile(r"-?\$?\s*[\d,]*\d(?:\.\d+)?")


def _normalize_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f"{int(f)}" if f.is_integer() else f"{f:.6g}"


def _gold_answer(answer_field: str) -> str | None:
    m = _ANSWER_RE.search(answer_field)
    return _normalize_number(m.group(1)) if m else None


def _extract_prediction(response: str) -> str | None:
    """Flexible-extract: last number in response wins."""
    matches = _NUMBER_RE.findall(response)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _score_all(rows: list[dict], responses: list[str]) -> AccResult:
    result = AccResult()
    for row, response in zip(rows, responses):
        gold = _gold_answer(row["answer"])
        pred = _extract_prediction(response)
        ok = gold is not None and pred is not None and gold == pred
        result.n += 1
        result.correct += int(ok)
        result.per_row.append(
            {
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "response": response,
            }
        )
    return result


def run_phase(
    handle: PhaseHandle,
    *,
    rows: list[dict],
    max_gen_tokens: int,
    max_concurrency: int,
) -> dict[str, Any]:
    batch_messages = [[{"role": "user", "content": r["question"]}] for r in rows]
    print(
        f"[gsm8k/{handle.phase}] firing {len(rows)} prompts "
        f"(max_concurrency={max_concurrency}, max_tokens={max_gen_tokens})",
        flush=True,
    )
    t0 = time.time()
    batch_out = complete_batch(
        handle,
        batch_messages,
        max_tokens=max_gen_tokens,
        temperature=0.0,
        max_concurrency=max_concurrency,
    )
    elapsed = time.time() - t0
    print(f"[gsm8k/{handle.phase}] gen done in {elapsed:.1f}s ({elapsed / len(rows):.2f}s avg)", flush=True)

    responses = [text for text, _ in batch_out]
    result = _score_all(rows, responses)
    summary = result.to_dict()
    summary["per_row"] = result.per_row
    return summary


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    n: int | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    max_gen_tokens: int = 1024,
    max_concurrency: int = 64,
) -> dict[str, Any]:
    ds = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows: list[dict] = [dict(r) for r in ds]

    return paired_eval(
        eval_name="gsm8k",
        rows=rows,
        ckpt=ckpt,
        output_dir=output_dir,
        run_phase_fn=run_phase,
        headline_metrics=["acc"],
        base_model=base_model,
        port=port,
        max_model_len=max_model_len,
        phase_kwargs={
            "max_gen_tokens": max_gen_tokens,
            "max_concurrency": max_concurrency,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="Truncate to first N rows (debug).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--max-concurrency", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt", "output_dir")
    run(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        base_model=args.base_model,
        n=args.n,
        port=args.port,
        max_model_len=args.max_model_len,
        max_gen_tokens=args.max_gen_tokens,
        max_concurrency=args.max_concurrency,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
