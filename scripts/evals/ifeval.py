"""Native IFEval — paired base vs ckpt.

Dataset: `HuggingFaceH4/ifeval` (541 prompts). Uses Google's canonical
IFEvalG verifiers (copied to `_ifeval_verifiers/`).

Metrics (4): prompt_level × strict/loose × inst_level × strict/loose.
- **prompt_level**: % of prompts where ALL instructions pass
- **inst_level**: % of individual instructions that pass
- **strict**: response as-generated
- **loose**: also try response variants (Google's
  `test_instruction_following_loose`: strip outer lines, strip `*`)

Output: `<output_dir>/ifeval.json` with base/ckpt/delta on each metric plus
per-row details.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ._ifeval_verifiers import instructions_registry
from ._server import PhaseHandle, complete_batch, paired_eval, resolve_path_args

IFEVAL_DATASET = "HuggingFaceH4/ifeval"
IFEVAL_SPLIT = "train"


# -------------------- loose transform --------------------


def _loose_candidates(response: str) -> list[str]:
    """Google's `test_instruction_following_loose` — try response variants."""
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    revised = response.replace("*", "")
    revised_first = remove_first.replace("*", "")
    revised_last = remove_last.replace("*", "")
    revised_both = remove_both.replace("*", "")
    return [
        response,
        revised,
        remove_first,
        revised_first,
        remove_last,
        revised_last,
        remove_both,
        revised_both,
    ]


# -------------------- verifier invocation --------------------


def _check_instruction(
    inst_id: str,
    inst_kwargs: dict,
    prompt: str,
    response: str,
    loose: bool,
) -> bool:
    """Instantiate the checker, build_description with dataset kwargs, then check."""
    cls = instructions_registry.INSTRUCTION_DICT.get(inst_id)
    if cls is None:
        return False
    # IFEval's `kwargs` dict has None for un-used constraint fields; drop them.
    filtered = {k: v for k, v in inst_kwargs.items() if v is not None}
    try:
        instr = cls(inst_id)
    except TypeError:
        instr = cls()
    try:
        args = instr.get_instruction_args() or {}
    except Exception:
        args = {}
    if args and "prompt" in args:
        filtered = dict(filtered)
        filtered.setdefault("prompt", prompt)
    try:
        instr.build_description(**filtered)
    except Exception:
        return False
    if not loose:
        try:
            return bool(instr.check_following(response))
        except Exception:
            return False
    # Loose: pass if ANY response variant passes.
    for candidate in _loose_candidates(response):
        if not candidate.strip():
            continue
        try:
            if instr.check_following(candidate):
                return True
        except Exception:
            continue
    return False


# -------------------- scoring --------------------


@dataclass
class Result:
    n_prompts: int = 0
    n_instructions: int = 0
    prompt_strict: int = 0
    prompt_loose: int = 0
    inst_strict: int = 0
    inst_loose: int = 0
    per_row: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_prompts": self.n_prompts,
            "n_instructions": self.n_instructions,
            "prompt_level_strict_acc": self.prompt_strict / self.n_prompts if self.n_prompts else 0.0,
            "prompt_level_loose_acc": self.prompt_loose / self.n_prompts if self.n_prompts else 0.0,
            "inst_level_strict_acc": self.inst_strict / self.n_instructions if self.n_instructions else 0.0,
            "inst_level_loose_acc": self.inst_loose / self.n_instructions if self.n_instructions else 0.0,
        }


def _score_all(
    rows: list[dict],
    responses: list[str],
) -> Result:
    result = Result()
    for row, response in zip(rows, responses):
        prompt = row["prompt"]
        inst_ids: list[str] = list(row["instruction_id_list"] or [])
        kwargs_list: list[dict] = list(row["kwargs"] or [{} for _ in inst_ids])

        strict_flags: list[bool] = []
        loose_flags: list[bool] = []
        for inst_id, inst_kwargs in zip(inst_ids, kwargs_list):
            s = _check_instruction(inst_id, inst_kwargs, prompt, response, loose=False)
            l = _check_instruction(inst_id, inst_kwargs, prompt, response, loose=True) or s
            strict_flags.append(s)
            loose_flags.append(l)

        result.n_prompts += 1
        result.n_instructions += len(inst_ids)
        if strict_flags and all(strict_flags):
            result.prompt_strict += 1
        if loose_flags and all(loose_flags):
            result.prompt_loose += 1
        result.inst_strict += sum(strict_flags)
        result.inst_loose += sum(loose_flags)

        result.per_row.append(
            {
                "key": row.get("key"),
                "instruction_id_list": inst_ids,
                "strict_flags": strict_flags,
                "loose_flags": loose_flags,
                "prompt_strict": bool(strict_flags and all(strict_flags)),
                "prompt_loose": bool(loose_flags and all(loose_flags)),
                "response": response,
            }
        )
    return result


# -------------------- per-phase runner --------------------


def run_phase(
    handle: PhaseHandle,
    *,
    rows: list[dict],
    max_gen_tokens: int,
    max_concurrency: int,
) -> dict[str, Any]:
    batch_messages = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    print(
        f"[ifeval/{handle.phase}] firing {len(rows)} prompts "
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
    print(f"[ifeval/{handle.phase}] gen done in {elapsed:.1f}s ({elapsed / len(rows):.2f}s avg)", flush=True)

    responses = [text for text, _ in batch_out]
    result = _score_all(rows, responses)
    summary = result.to_dict()
    summary["per_row"] = result.per_row
    return summary


# -------------------- top-level --------------------


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    n: int | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    max_gen_tokens: int = 1280,
    max_concurrency: int = 64,
) -> dict[str, Any]:
    ds = load_dataset(IFEVAL_DATASET, split=IFEVAL_SPLIT)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows: list[dict] = [dict(r) for r in ds]

    return paired_eval(
        eval_name="ifeval",
        rows=rows,
        ckpt=ckpt,
        output_dir=output_dir,
        run_phase_fn=run_phase,
        headline_metrics=[
            "prompt_level_strict_acc",
            "prompt_level_loose_acc",
            "inst_level_strict_acc",
            "inst_level_loose_acc",
        ],
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
    parser.add_argument("--max-gen-tokens", type=int, default=1280)
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
