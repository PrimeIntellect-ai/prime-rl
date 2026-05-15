"""Native MMLU:rc — paired base vs ckpt, per-char log-likelihood.

Dataset: `cais/mmlu` config `all` split `test` (14042 rows × 4 choices).
Metric: ranked-classification (`:rc`). Each of the 4 choices is scored by
per-character average log-likelihood of the choice text given the context,
then argmax. Length-invariant, unlike the `:mc` single-letter approach.

Protocol: single prompt-string per (question, choice); sent via
`/v1/completions` with `echo=True, logprobs=0, max_tokens=1` so vLLM returns
per-token log-probs for all prompt tokens (plus one trailing generated token
we ignore). Continuation tokens are located via `text_offset` — the first
token whose character-offset ≥ len(context) starts the continuation.

Neither phase uses chat templates; `:rc` compares raw-string likelihoods so
formatting consistency beats instruction formatting.

Output: `<output_dir>/mmlu.json` with base/ckpt/delta accuracy + per-row
details.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset

from ._server import AccResult, PhaseHandle, paired_eval, resolve_path_args

MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"


def _format_context(row: dict) -> str:
    """Canonical MMLU preamble — matches lm-eval-harness / olmes."""
    subject = row.get("subject", "general knowledge").replace("_", " ")
    choices = row["choices"]
    letters = ["A", "B", "C", "D"]
    lines = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
    lines.append(row["question"])
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


_RETRY_DELAYS: tuple[float, ...] = (0.0, 1.0, 4.0, 16.0)  # 4 attempts: immediate, 1s, 4s, 16s


async def _post_completion(
    base_url: str,
    api_key: str,
    model_id: str,
    full: str,
    client: httpx.AsyncClient,
) -> httpx.Response:
    resp = await client.post(
        f"{base_url}/v1/completions",
        json={
            "model": model_id,
            "prompt": full,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 0,
            "temperature": 0.0,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    )
    # 4xx is a client bug — surface immediately, no retry.
    if 400 <= resp.status_code < 500:
        resp.raise_for_status()
    return resp


async def _call_with_retry(
    base_url: str,
    api_key: str,
    model_id: str,
    full: str,
    client: httpx.AsyncClient,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            resp = await _post_completion(base_url, api_key, model_id, full, client)
            # Retry only 5xx (transient server errors).
            if resp.status_code >= 500:
                err = httpx.HTTPStatusError(f"server error {resp.status_code}", request=resp.request, response=resp)
                if attempt == len(_RETRY_DELAYS) - 1:
                    raise err
                last_exc = err
                print(
                    f"[mmlu] retry {attempt + 1}/{len(_RETRY_DELAYS) - 1} after 5xx: {err}",
                    flush=True,
                )
                continue
            return resp
        except httpx.RequestError as e:
            # Connection/timeout/transport — transient, retry-eligible.
            if attempt == len(_RETRY_DELAYS) - 1:
                raise
            last_exc = e
            print(
                f"[mmlu] retry {attempt + 1}/{len(_RETRY_DELAYS) - 1} after {type(e).__name__}: {e}",
                flush=True,
            )
    # Unreachable: loop either returns or raises on final attempt.
    raise RuntimeError(f"_call_with_retry exhausted without return; last_exc={last_exc}")


async def _score_continuation(
    base_url: str,
    api_key: str,
    model_id: str,
    context: str,
    continuation: str,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
) -> float:
    """Per-char average log-probability of `continuation` given `context`."""
    full = context + continuation
    async with semaphore:
        resp = await _call_with_retry(base_url, api_key, model_id, full, client)
    resp.raise_for_status()
    data = resp.json()
    lp = data["choices"][0].get("logprobs")
    if lp is None:
        raise RuntimeError("vLLM did not return logprobs — expected echo=True, logprobs=0")
    token_logprobs: list[float | None] = lp["token_logprobs"]
    text_offset: list[int] = lp["text_offset"]
    ctx_len = len(context)
    # `text_offset[i]` = character offset where token i starts in the returned text.
    start_idx = next((i for i, o in enumerate(text_offset) if o >= ctx_len), None)
    if start_idx is None or start_idx >= len(token_logprobs):
        return float("-inf")
    # Last token corresponds to the max_tokens=1 trailing generation; exclude it.
    end_idx = len(token_logprobs) - 1
    vals = [v for v in token_logprobs[start_idx:end_idx] if v is not None]
    if not vals:
        return float("-inf")
    # Per-character normalization (length-invariant `:rc`).
    return sum(vals) / max(len(continuation), 1)


async def _score_all_async(
    rows: list[dict],
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    max_concurrency: int,
) -> AccResult:
    semaphore = asyncio.Semaphore(max_concurrency)
    contexts = [_format_context(r) for r in rows]
    # Continuation: leading space + choice text (per lm-eval convention).
    jobs: list[tuple[int, int, str, str]] = []  # (row_idx, choice_idx, context, continuation)
    for ri, (row, ctx) in enumerate(zip(rows, contexts)):
        for ci, choice_text in enumerate(row["choices"]):
            jobs.append((ri, ci, ctx, " " + choice_text))

    async with httpx.AsyncClient(http2=False) as client:
        tasks = [
            _score_continuation(base_url, api_key, model_id, ctx, cont, semaphore, client) for (_, _, ctx, cont) in jobs
        ]
        # Fail-fast: any unrecoverable error (post-retry) propagates up.
        scores = await asyncio.gather(*tasks)

    # Reshape: [n_rows][4]
    per_row_scores: list[list[float]] = [[float("-inf")] * 4 for _ in rows]
    for (ri, ci, _, _), s in zip(jobs, scores):
        per_row_scores[ri][ci] = s

    # Defensive: if any row is all -inf across its 4 choices, argmax is meaningless.
    # With retries+fail-fast above, the only path here is a legitimate all-token-None
    # logprobs response, which is a contract violation worth surfacing loudly.
    for ri, choice_scores in enumerate(per_row_scores):
        if all(s == float("-inf") for s in choice_scores):
            raise RuntimeError(
                f"[mmlu] row={ri} has all-(-inf) scores across 4 choices; "
                f"cannot compute argmax. Check vLLM logprobs response contract."
            )

    result = AccResult()
    for ri, (row, choice_scores) in enumerate(zip(rows, per_row_scores)):
        pred = max(range(4), key=lambda i: choice_scores[i])
        gold = int(row["answer"])
        ok = pred == gold
        result.n += 1
        result.correct += int(ok)
        result.per_row.append(
            {
                "subject": row.get("subject"),
                "question": row["question"],
                "choices": row["choices"],
                "gold": gold,
                "pred": pred,
                "scores": choice_scores,
                "correct": ok,
            }
        )
    return result


def run_phase(
    handle: PhaseHandle,
    *,
    rows: list[dict],
    max_concurrency: int,
) -> dict[str, Any]:
    print(
        f"[mmlu/{handle.phase}] firing {len(rows)}×4 likelihood queries (max_concurrency={max_concurrency})",
        flush=True,
    )
    t0 = time.time()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("mmlu.run_phase called from running event loop")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        _score_all_async(
            rows,
            base_url=handle.base_url,
            api_key=handle.api_key,
            model_id=handle.model_id,
            max_concurrency=max_concurrency,
        )
    )
    elapsed = time.time() - t0
    print(f"[mmlu/{handle.phase}] scoring done in {elapsed:.1f}s", flush=True)

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
    max_concurrency: int = 64,
) -> dict[str, Any]:
    ds = load_dataset(MMLU_DATASET, MMLU_CONFIG, split=MMLU_SPLIT)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows: list[dict] = [dict(r) for r in ds]

    return paired_eval(
        eval_name="mmlu",
        rows=rows,
        ckpt=ckpt,
        output_dir=output_dir,
        run_phase_fn=run_phase,
        headline_metrics=["acc"],
        base_model=base_model,
        port=port,
        max_model_len=max_model_len,
        phase_kwargs={"max_concurrency": max_concurrency},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="Truncate to first N rows (debug).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
        max_concurrency=args.max_concurrency,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
