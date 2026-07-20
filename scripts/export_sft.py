"""Export a run's saved traces (traces.jsonl) as an SFT dataset for `uv run sft`.

Reads either traces.jsonl shape and reshapes it into the dataset shape the SFT trainer
consumes directly (see `prime_rl.trainer.sft.data`): a `messages` column (OpenAI chat wire
shape) plus a `tools` column (the tools the model was shown, from `Trace.tools`,
JSON-encoded — heterogeneous JSON-schema dicts don't fit a fixed Arrow schema). One row per
branch: a linear rollout contributes one sample, a compacted/subagent rollout one per branch
(one training sample is built per branch). Each line is sniffed individually: a verifiers v1
eval run writes one *episode* per line, while prime-rl's rollout dumps
(`rollouts/step_N/.../traces.jsonl`) and pre-episode eval runs write one flat *trace* per line.

Selection: generation-errored traces (`stop_condition == "error"`) always drop — a broken
transcript is not a sample. A scoring-only error keeps the generation outcome as its stop
condition and a complete conversation, so it stays; its reward may be partial/zero, which
`--min-reward` handles. Untrainable traces (a multi-agent episode's frozen seats — a judge,
a pinned user sim) drop by default; `--include-untrainable` keeps them.

Usage (from the prime-rl repo):
    uv run python scripts/export_sft.py <run-dir> [--min-reward 1.0] [--drop-truncated]
                                        [--include-untrainable] [-o OUT_DIR] [--push HF_REPO_ID]

Writes `<run-dir>/sft/train.parquet` by default — point the trainer at it with
`--data.name <run-dir>/sft`. Requires a verifiers release carrying `Trace.tools`
(PrimeIntellect-ai/verifiers#1963).
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset
from pydantic import ValidationError
from verifiers.v1 import Trace, WireEpisode, WireTrace
from verifiers.v1.dialects.chat import message_to_wire


def sft_rows(trace: Trace) -> list[dict]:
    """A trace's SFT rows — one per branch: the branch's conversation as OpenAI chat wire
    dicts plus the trace's advertised tools, JSON-encoded."""
    tools = json.dumps([t.model_dump(mode="json", exclude_none=True) for t in trace.tools or []])
    return [
        {
            "messages": [message_to_wire(m) for m in branch.messages],
            "tools": tools,
        }
        for branch in trace.branches
        if branch.messages
    ]


def keep(trace: Trace, min_reward: float | None, drop_truncated: bool) -> bool:
    """Whether a trace is worth training on (see module docstring for the error semantics)."""
    if trace.stop_condition == "error":
        return False
    if drop_truncated and trace.is_truncated:
        return False
    return min_reward is None or trace.reward >= min_reward


def iter_traces(traces_path: Path) -> Iterator[Trace]:
    """Yield every trace in a traces.jsonl, sniffing each line's shape: a dict with a
    "traces" key is an episode (the env-rollout atom, its traces carry the chats),
    anything else a single flat trace record."""
    with traces_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict) and "traces" in record:
                    yield from WireEpisode.model_validate(record).traces
                else:
                    yield WireTrace.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as e:
                raise SystemExit(f"export-sft: {traces_path}:{lineno}: not an episode or trace record: {e}") from e


def collect_rows(
    traces_path: Path, *, min_reward: float | None, drop_truncated: bool, include_untrainable: bool
) -> list[dict]:
    """The selected traces' SFT rows (see module docstring for the selection semantics)."""
    total, untrainable, rows = 0, 0, []
    for trace in iter_traces(traces_path):
        total += 1
        if not trace.trainable and not include_untrainable:
            untrainable += 1
            continue
        if keep(trace, min_reward, drop_truncated):
            rows.extend(sft_rows(trace))
    if untrainable:
        print(f"export-sft: dropped {untrainable} untrainable trace(s) — pass --include-untrainable to keep them")
    print(f"export-sft: {total} trace(s) -> {len(rows)} row(s)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the run dir (holds traces.jsonl)")
    parser.add_argument("--min-reward", type=float, default=None, help="keep traces with reward >= this")
    parser.add_argument("--drop-truncated", action="store_true", help="drop budget-cut traces")
    parser.add_argument(
        "--include-untrainable", action="store_true", help="keep frozen-seat (trainable=False) traces too"
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="default: <run-dir>/sft")
    parser.add_argument("--push", default=None, help="HF repo id to push to instead of writing parquet")
    args = parser.parse_args()

    traces_path = args.run_dir / "traces.jsonl"
    if not traces_path.exists():
        raise SystemExit(f"no traces.jsonl in {args.run_dir}")

    rows = collect_rows(
        traces_path,
        min_reward=args.min_reward,
        drop_truncated=args.drop_truncated,
        include_untrainable=args.include_untrainable,
    )
    if not rows:
        raise SystemExit("export-sft: no rows to export after selection")

    dataset = Dataset.from_list(rows)
    if args.push:
        dataset.push_to_hub(args.push)
        print(f"export-sft: pushed to {args.push}")
        return
    out = args.output_dir or args.run_dir / "sft"
    out.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(out / "train.parquet"))
    print(f"export-sft: wrote {out / 'train.parquet'} -> train with --data.name {out}")


if __name__ == "__main__":
    main()
