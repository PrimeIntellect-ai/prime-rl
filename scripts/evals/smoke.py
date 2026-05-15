"""Qualitative base-vs-ckpt smoke eval using Protocol B.

Protocol B (empirically verified 2026-04-19 on marin-8b, see session log):
  - Render each prompt client-side with the **SFT tokenizer's** chat template.
  - Send the rendered string via `/v1/completions` (raw prompt) to vLLM.
  - Both base and ckpt weights see byte-identical token sequences — the only
    variable is weights.

This replaces the older `client.completions.create(prompt=instruction_wrapper)`
approach which caused the base marin-8b to hallucinate the wrapper text and
loop. See scripts/evals/_server.py docstring for Protocol B rationale.

Outputs:
  <output_dir>/results.json  — per-prompt, both phases, both responses
  <output_dir>/comparison.md — side-by-side markdown for human review
  <output_dir>/aggregate.json — length + paragraph stats for base/ckpt/delta
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ._server import PhaseHandle, complete_batch, infer_base_model, load_sft_tokenizer, paired_run, resolve_path_args

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "smoke_eval_sft_prompts.json"


def _load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt suite at {path} must be a non-empty JSON list.")
    return prompts


def _completion_metrics(text: str, usage: Any) -> dict[str, Any]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(lines),
        "paragraphs": len(paragraphs),
        "bullets": sum(line.lstrip().startswith(("-", "*")) for line in lines),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def run_phase(
    handle: PhaseHandle,
    *,
    prompts: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_concurrency: int = 64,
) -> list[dict[str, Any]]:
    """Run the prompt suite against a single phase (base or ckpt) in one batched call."""
    batch_messages = [[{"role": "user", "content": p["prompt"]}] for p in prompts]
    # Use the largest max_tokens across per-prompt overrides to stay uniform in-batch.
    effective_max_tokens = max(int(p.get("max_tokens", max_tokens)) for p in prompts)

    print(
        f"[smoke/{handle.phase}] firing {len(prompts)} prompts (max_concurrency={max_concurrency}, "
        f"max_tokens={effective_max_tokens})",
        flush=True,
    )
    t0 = time.perf_counter()
    batch_results = complete_batch(
        handle,
        batch_messages,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_concurrency=max_concurrency,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[smoke/{handle.phase}] batch done in {elapsed:.1f}s (= {elapsed / len(prompts):.1f}s avg/prompt wall)",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for prompt, (text, usage) in zip(prompts, batch_results):
        metrics = _completion_metrics(text, usage)
        results.append(
            {
                "phase": handle.phase,
                "id": prompt["id"],
                "source": prompt["source"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "response": text,
                "metrics": metrics,
            }
        )
    return results


def _aggregate(results: list[dict]) -> dict[str, Any]:
    token_counts = [r["metrics"]["completion_tokens"] for r in results if r["metrics"]["completion_tokens"] is not None]
    paragraphs = [r["metrics"]["paragraphs"] for r in results]
    return {
        "n": len(results),
        "median_tokens": statistics.median(token_counts) if token_counts else None,
        "p90_tokens": (statistics.quantiles(token_counts, n=10)[-1] if len(token_counts) >= 10 else None),
        "median_paragraphs": statistics.median(paragraphs) if paragraphs else None,
        "empty_responses": sum(1 for r in results if not r["response"].strip()),
    }


def _write_markdown(
    output_path: Path,
    *,
    base_model: str,
    ckpt: Path,
    prompts_path: Path,
    sft_tokenizer_name: str,
    model_id: str,
    base_results: list[dict],
    ckpt_results: list[dict],
) -> None:
    ckpt_by_id = {r["id"]: r for r in ckpt_results}
    lines = [
        "# SFT Smoke Eval — base vs ckpt (Protocol B)",
        "",
        f"- Base weights:  `{base_model}`",
        f"- Checkpoint:    `{ckpt}`",
        f"- Served id:     `{model_id}`",
        f"- SFT tokenizer: `{sft_tokenizer_name}` (used to render prompts for BOTH phases)",
        f"- Prompt suite:  `{prompts_path}`",
        "",
        "## Summary",
        "",
        "| Prompt | Source | Base tok | Ckpt tok |",
        "| --- | --- | ---: | ---: |",
    ]
    for base_item in base_results:
        ci = ckpt_by_id[base_item["id"]]
        lines.append(
            f"| {base_item['id']} | {base_item['source']} | "
            f"{base_item['metrics']['completion_tokens'] or '?'} | "
            f"{ci['metrics']['completion_tokens'] or '?'} |"
        )
    for base_item in base_results:
        ci = ckpt_by_id[base_item["id"]]
        lines += [
            "",
            f"## {base_item['id']}",
            "",
            f"Source: `{base_item['source']}`",
            "",
            "### Prompt",
            "",
            base_item["prompt"],
            "",
            "### Base",
            "",
            f"Metrics: `{json.dumps(base_item['metrics'], sort_keys=True)}`",
            "",
            base_item["response"],
            "",
            "### Checkpoint",
            "",
            f"Metrics: `{json.dumps(ci['metrics'], sort_keys=True)}`",
            "",
            ci["response"],
            "",
        ]
    output_path.write_text("\n".join(lines))


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    prompts_path: Path | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    max_tokens: int = 2400,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """End-to-end paired base/ckpt smoke eval.

    Boots vLLM on base, runs the 14-prompt suite (base phase), hot-reloads ckpt,
    runs the suite again (ckpt phase). Writes results.json, comparison.md,
    aggregate.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = prompts_path or DEFAULT_PROMPTS_PATH
    prompts = _load_prompts(prompts_path)
    base = base_model or infer_base_model(ckpt)
    sft_tokenizer = load_sft_tokenizer(ckpt)
    log_path = output_dir / "inference.log"

    with paired_run(
        base_model=base,
        ckpt_dir=ckpt,
        sft_tokenizer=sft_tokenizer,
        port=port,
        max_model_len=max_model_len,
        log_path=log_path,
    ) as (enter_base, enter_ckpt):
        base_handle = enter_base()
        print(f"\n[smoke] BASE pass (server model_id={base_handle.model_id})\n", flush=True)
        base_results = run_phase(
            base_handle,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        ckpt_handle = enter_ckpt()
        print(f"\n[smoke] CKPT pass (weights swapped to {ckpt})\n", flush=True)
        ckpt_results = run_phase(
            ckpt_handle,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    base_agg = _aggregate(base_results)
    ckpt_agg = _aggregate(ckpt_results)
    delta = {
        k: (ckpt_agg.get(k) - base_agg.get(k))
        if isinstance(ckpt_agg.get(k), (int, float)) and isinstance(base_agg.get(k), (int, float))
        else None
        for k in base_agg
    }

    payload = {
        "base_model": base,
        "ckpt": str(ckpt),
        "sft_tokenizer": getattr(sft_tokenizer, "name_or_path", "unknown"),
        "served_model_id": base_handle.model_id,
        "prompts_path": str(prompts_path),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "base_results": base_results,
        "ckpt_results": ckpt_results,
        "aggregate": {"base": base_agg, "ckpt": ckpt_agg, "delta": delta},
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "aggregate.json").write_text(json.dumps(payload["aggregate"], indent=2))
    _write_markdown(
        output_dir / "comparison.md",
        base_model=base,
        ckpt=ckpt,
        prompts_path=prompts_path,
        sft_tokenizer_name=getattr(sft_tokenizer, "name_or_path", "unknown"),
        model_id=base_handle.model_id,
        base_results=base_results,
        ckpt_results=ckpt_results,
    )
    print(
        f"\n[smoke] wrote {output_dir}/results.json, comparison.md, aggregate.json\n",
        flush=True,
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt", "output_dir")
    run(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        base_model=args.base_model,
        prompts_path=args.prompts,
        port=args.port,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
