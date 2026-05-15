"""
Headroom probe for zebra-puzzles-v2 diagnostic pair.

Measures base pass@1 and pass@8 of Qwen/Qwen3-4B-Instruct-2507 on MCQ
across grid_size buckets, using the Chen et al. (2021) unbiased estimator.

Goal: pick the Goldilocks grid_size bucket where base pass@1 ∈ [35%, 75%] —
enough signal for GRPO advantages to be non-zero, not saturated.

Run via `sbatch tmp/zebra_probe/probe.sbatch` on Isambard-AI Phase 2.
Target hardware: GH200 (sm_90 Hopper) → bf16 only, no fp8.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
GRIDS = ["3x3", "4x4"]
N_PUZZLES_PER_GRID = 200
N_SAMPLES = 8
TEMPERATURE = 0.7
MAX_TOKENS = 512
OUT = Path("tmp/zebra_probe/pass_at_n_results.json")


SYSTEM = (
    "You are solving a logic puzzle. "
    "Reason step-by-step, then output your final choice on the very last line "
    "as exactly one of: Answer: A, Answer: B, Answer: C, Answer: D, Answer: E, or Answer: F."
)

ANSWER_RE = re.compile(r"Answer:\s*([A-F])", re.IGNORECASE)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def parse_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None


def build_chat_prompt(prompt_mcq: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt_mcq},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True, help="path to pre-staged train_canonical parquet shard")
    args = ap.parse_args()

    df = pq.read_table(
        args.parquet,
        columns=[
            "puzzle_id",
            "grid_size",
            "prompt_mcq",
            "choices",
            "target_label",
        ],
    ).to_pandas()

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        stop=None,
    )

    results = {}
    for grid in GRIDS:
        sub = df[df["grid_size"] == grid].head(N_PUZZLES_PER_GRID).reset_index(drop=True)
        prompts = [build_chat_prompt(p, tok) for p in sub["prompt_mcq"]]
        truths = sub["target_label"].tolist()

        print(f"\n=== {grid} — {len(prompts)} puzzles × {N_SAMPLES} samples @ T={TEMPERATURE} ===")
        outs = llm.generate(prompts, sp)

        per_puzzle_c = []
        parse_failures = 0
        label_hist = Counter()
        for out, truth in zip(outs, truths):
            c = 0
            for completion in out.outputs:
                ans = parse_answer(completion.text)
                if ans is None:
                    parse_failures += 1
                    continue
                label_hist[ans] += 1
                if ans == truth:
                    c += 1
            per_puzzle_c.append(c)

        pass_1 = sum(pass_at_k(N_SAMPLES, c, 1) for c in per_puzzle_c) / len(per_puzzle_c)
        pass_8 = sum(pass_at_k(N_SAMPLES, c, N_SAMPLES) for c in per_puzzle_c) / len(per_puzzle_c)
        solve_none = sum(1 for c in per_puzzle_c if c == 0) / len(per_puzzle_c)
        solve_all = sum(1 for c in per_puzzle_c if c == N_SAMPLES) / len(per_puzzle_c)
        parse_rate = 1 - parse_failures / (len(prompts) * N_SAMPLES)

        print(f"  pass@1 = {pass_1:.3f}   pass@8 = {pass_8:.3f}")
        print(
            f"  solve_none = {solve_none:.3f}   solve_all = {solve_all:.3f}   "
            f"GRPO effective = {1 - solve_none - solve_all:.3f}"
        )
        print(f"  parse_rate = {parse_rate:.3f}   label hist = {dict(label_hist)}")

        results[grid] = {
            "n_puzzles": len(prompts),
            "n_samples": N_SAMPLES,
            "temperature": TEMPERATURE,
            "pass_at_1": pass_1,
            "pass_at_8": pass_8,
            "solve_none": solve_none,
            "solve_all": solve_all,
            "grpo_effective": 1 - solve_none - solve_all,
            "parse_rate": parse_rate,
            "label_hist": dict(label_hist),
            "per_puzzle_c": per_puzzle_c,
        }

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== Goldilocks verdict ===")
    for grid, r in results.items():
        p1 = r["pass_at_1"]
        eff = r["grpo_effective"]
        if 0.35 <= p1 <= 0.75 and eff >= 0.3:
            verdict = "GOLDILOCKS (use this)"
        elif p1 > 0.75:
            verdict = "too easy (saturated, low headroom)"
        elif p1 < 0.35:
            verdict = "too hard (near-floor, Cromwell's rule risk)"
        else:
            verdict = "marginal (low GRPO effective batch)"
        print(f"  {grid}: pass@1={p1:.2f}  effective={eff:.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
