"""MT-Bench (80 prompts, 8 categories, 2-turn) — paired base vs ckpt via Protocol B.

Protocol B: prompts rendered client-side with the SFT chat template, sent via
`/v1/completions`. Same protocol for base and ckpt so the only variable is
weights. See `_server.py` for rationale.

Judge: Claude-Sonnet-4-6, frozen per §10a. Canonical FastChat templates:
  - `single-v1-multi-turn` (general)
  - `single-math-v1-multi-turn` (math, reasoning, coding — GPT-4 reference answers)

Output: `<output_dir>/mtbench.json` with per-phase `overall`, `per_category`,
and per-question `judgments` including base/ckpt answer pairs side-by-side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from ._judge import Judge, JudgeCall
from ._server import PhaseHandle, complete_batch, infer_base_model, load_sft_tokenizer, paired_run, resolve_path_args

MTBENCH_CACHE_ROOT = Path("outputs/evals/_cache/mtbench")
QUESTION_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
JUDGE_PROMPTS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/judge_prompts.jsonl"
REF_ANSWERS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"

NEED_REF_CATS = {"math", "reasoning", "coding"}
RATING_RE = re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)


def _load_jsonl(path: Path, url: str) -> list[dict]:
    if not path.exists():
        print(f"[mtbench] downloading {url} → {path}", flush=True)
        _download(url, path)
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_questions(cache_root: Path = MTBENCH_CACHE_ROOT) -> list[dict]:
    return _load_jsonl(cache_root / "question.jsonl", QUESTION_URL)


def load_judge_templates(cache_root: Path = MTBENCH_CACHE_ROOT) -> dict[str, dict]:
    rows = _load_jsonl(cache_root / "judge_prompts.jsonl", JUDGE_PROMPTS_URL)
    return {row["name"]: row for row in rows}


def load_references(cache_root: Path = MTBENCH_CACHE_ROOT) -> dict[int, list[str]]:
    rows = _load_jsonl(cache_root / "reference_gpt4.jsonl", REF_ANSWERS_URL)
    out: dict[int, list[str]] = {}
    for row in rows:
        qid = row["question_id"]
        turns = row["choices"][0]["turns"]
        out[qid] = turns
    return out


# -------------------- rollout --------------------


def run_rollouts_batched(
    handle: PhaseHandle,
    turns_list: list[list[str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_concurrency: int = 64,
) -> list[list[str]]:
    """Batched 2-turn (or N-turn) rollouts. Fires all turn-k prompts concurrently,
    waits, threads responses into turn-(k+1) prompts, fires again.

    Assumes all conversations have the same number of turns (2 for MT-Bench).
    Returns a list-of-lists: responses_by_conv[i] = [turn1_resp, turn2_resp, ...].
    """
    n = len(turns_list)
    if n == 0:
        return []
    n_turns = len(turns_list[0])
    if not all(len(t) == n_turns for t in turns_list):
        raise ValueError("all conversations must have the same number of turns")

    # Running message state per conversation.
    conv_messages: list[list[dict[str, str]]] = [[] for _ in range(n)]
    responses: list[list[str]] = [[] for _ in range(n)]

    for t_idx in range(n_turns):
        for i in range(n):
            conv_messages[i].append({"role": "user", "content": turns_list[i][t_idx]})
        print(f"[mtbench/{handle.phase}] turn-{t_idx + 1} batch N={n}...", flush=True)
        t0 = time.time()
        turn_out = complete_batch(
            handle,
            conv_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_concurrency=max_concurrency,
        )
        print(f"[mtbench/{handle.phase}] turn-{t_idx + 1} done in {time.time() - t0:.1f}s", flush=True)
        for i, (text, _) in enumerate(turn_out):
            conv_messages[i].append({"role": "assistant", "content": text})
            responses[i].append(text)
    return responses


# -------------------- judging --------------------


def _parse_rating(text: str) -> float | None:
    match = RATING_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if 1.0 <= value <= 10.0:
        return value
    return None


def _build_judge_call(
    template: dict,
    *,
    question_1: str,
    answer_1: str,
    question_2: str,
    answer_2: str,
    ref_answer_1: str | None = None,
    ref_answer_2: str | None = None,
) -> JudgeCall:
    fill: dict[str, str] = {
        "question_1": question_1,
        "answer_1": answer_1,
        "question_2": question_2,
        "answer_2": answer_2,
    }
    if ref_answer_1 is not None:
        fill["ref_answer_1"] = ref_answer_1
    if ref_answer_2 is not None:
        fill["ref_answer_2"] = ref_answer_2
    return JudgeCall(
        prompt=template["prompt_template"].format(**fill),
        system=template.get("system_prompt"),
        max_tokens=1024,
        temperature=0.0,
    )


def _category_template(templates: dict[str, dict], category: str) -> dict:
    key = "single-math-v1-multi-turn" if category in NEED_REF_CATS else "single-v1-multi-turn"
    if key not in templates:
        raise KeyError(f"MT-Bench judge template '{key}' missing")
    return templates[key]


# -------------------- per-phase runner --------------------


def _generate_rollouts(
    handle: PhaseHandle,
    questions: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    max_concurrency: int = 64,
) -> list[dict[str, Any]]:
    turns_list = [q["turns"] for q in questions]
    all_responses = run_rollouts_batched(
        handle,
        turns_list,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrency=max_concurrency,
    )
    return [
        {
            "question_id": q["question_id"],
            "category": q["category"],
            "turns": q["turns"],
            "answers": answers,
        }
        for q, answers in zip(questions, all_responses)
    ]


def _judge_rollouts(
    rollouts: list[dict],
    *,
    judge: Judge,
    templates: dict,
    references: dict[int, list[str]],
    max_concurrency: int = 16,
) -> list[dict[str, Any]]:
    """Judge all rollouts concurrently (AsyncAnthropic + semaphore)."""
    # Build calls in order, remember which ones are skippable (<no-rollout>).
    calls: list[JudgeCall | None] = []
    for r in rollouts:
        if len(r["answers"]) != 2:
            calls.append(None)
            continue
        template = _category_template(templates, r["category"])
        ref_turns = references.get(r["question_id"])
        ref_1 = ref_turns[0] if ref_turns and r["category"] in NEED_REF_CATS else None
        ref_2 = ref_turns[1] if ref_turns and r["category"] in NEED_REF_CATS and len(ref_turns) > 1 else None
        calls.append(
            _build_judge_call(
                template,
                question_1=r["turns"][0],
                answer_1=r["answers"][0],
                question_2=r["turns"][1],
                answer_2=r["answers"][1],
                ref_answer_1=ref_1,
                ref_answer_2=ref_2,
            )
        )

    # Only send valid calls to the batch; slot in None responses for skipped.
    valid_idxs = [i for i, c in enumerate(calls) if c is not None]
    valid_calls = [calls[i] for i in valid_idxs]
    print(
        f"[mtbench/judge] firing {len(valid_calls)} concurrent judge calls (max_concurrency={max_concurrency})",
        flush=True,
    )
    t0 = time.time()
    valid_responses = judge.call_batch(valid_calls, max_concurrency=max_concurrency) if valid_calls else []
    print(f"[mtbench/judge] done in {time.time() - t0:.1f}s", flush=True)

    responses: list[Any] = [None] * len(rollouts)
    for i, resp in zip(valid_idxs, valid_responses):
        responses[i] = resp

    out: list[dict[str, Any]] = []
    for r, resp in zip(rollouts, responses):
        if resp is None:
            if len(r["answers"]) != 2:
                out.append({**r, "rating": None, "rating_text": "<no-rollout>"})
            else:
                out.append({**r, "rating": None, "rating_text": "<judge-error>"})
            continue
        rating = _parse_rating(resp.text)
        out.append({**r, "rating": rating, "rating_text": resp.text})
    return out


def _aggregate(judgments: list[dict]) -> dict[str, Any]:
    ratings = [j["rating"] for j in judgments if j["rating"] is not None]
    per_cat: dict[str, list[float]] = {}
    for j in judgments:
        if j["rating"] is not None:
            per_cat.setdefault(j["category"], []).append(j["rating"])
    return {
        "overall": sum(ratings) / len(ratings) if ratings else None,
        "per_category": {cat: sum(vs) / len(vs) for cat, vs in per_cat.items()},
        "parse_fail_count": sum(1 for j in judgments if j["rating"] is None),
    }


# -------------------- top-level --------------------


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    n: int | None = None,
    seed: int = 42,
    port: int = 8000,
    max_model_len: int = 4096,
    max_gen_tokens: int = 1024,
    gen_temperature: float = 0.7,
    judge_model: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    if n is not None:
        questions = questions[:n]
    templates = load_judge_templates()
    references = load_references()
    base = base_model or infer_base_model(ckpt)
    sft_tokenizer = load_sft_tokenizer(ckpt)
    judge_kwargs: dict[str, Any] = {}
    if judge_model is not None:
        judge_kwargs["model"] = judge_model
    judge = Judge(**judge_kwargs)

    log_path = output_dir / "inference.log"
    with paired_run(
        base_model=base,
        ckpt_dir=ckpt,
        sft_tokenizer=sft_tokenizer,
        port=port,
        max_model_len=max_model_len,
        log_path=log_path,
    ) as (enter_base, enter_ckpt):
        t0 = time.time()

        base_handle = enter_base()
        print(f"\n[mtbench] BASE rollouts (model_id={base_handle.model_id})\n", flush=True)
        base_rollouts = _generate_rollouts(
            base_handle,
            questions,
            max_tokens=max_gen_tokens,
            temperature=gen_temperature,
        )

        ckpt_handle = enter_ckpt()
        print("\n[mtbench] CKPT rollouts\n", flush=True)
        ckpt_rollouts = _generate_rollouts(
            ckpt_handle,
            questions,
            max_tokens=max_gen_tokens,
            temperature=gen_temperature,
        )
        rollout_elapsed = time.time() - t0

    # Judging happens after the server is torn down — it only calls Anthropic.
    print(f"\n[mtbench] judging base rollouts ({len(base_rollouts)})...\n", flush=True)
    base_judgments = _judge_rollouts(base_rollouts, judge=judge, templates=templates, references=references)
    print(f"\n[mtbench] judging ckpt rollouts ({len(ckpt_rollouts)})...\n", flush=True)
    ckpt_judgments = _judge_rollouts(ckpt_rollouts, judge=judge, templates=templates, references=references)

    base_agg = _aggregate(base_judgments)
    ckpt_agg = _aggregate(ckpt_judgments)
    delta_overall: float | None = None
    if base_agg["overall"] is not None and ckpt_agg["overall"] is not None:
        delta_overall = ckpt_agg["overall"] - base_agg["overall"]

    summary = {
        "base_model": base,
        "ckpt": str(ckpt),
        "sft_tokenizer": getattr(sft_tokenizer, "name_or_path", "unknown"),
        "judge_model": judge.model,
        "seed": seed,
        "n": len(questions),
        "phases": {
            "base": {"judgments": base_judgments, "aggregate": base_agg},
            "ckpt": {"judgments": ckpt_judgments, "aggregate": ckpt_agg},
        },
        "delta_overall": delta_overall,
        "rollout_elapsed_s": rollout_elapsed,
    }
    (output_dir / "mtbench.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[mtbench] wrote {output_dir}/mtbench.json", flush=True)
    print(
        json.dumps(
            {
                "base_overall": base_agg["overall"],
                "ckpt_overall": ckpt_agg["overall"],
                "delta_overall": delta_overall,
                "base_per_category": base_agg["per_category"],
                "ckpt_per_category": ckpt_agg["per_category"],
            },
            indent=2,
        )
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="Truncate to first N questions (debug).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--judge-model", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt", "output_dir")
    run(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        base_model=args.base_model,
        n=args.n,
        seed=args.seed,
        port=args.port,
        max_model_len=args.max_model_len,
        max_gen_tokens=args.max_gen_tokens,
        gen_temperature=args.temperature,
        judge_model=args.judge_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
