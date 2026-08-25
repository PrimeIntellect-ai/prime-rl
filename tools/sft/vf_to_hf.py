"""Convert a run's episodes (traces.jsonl) into an HF SFT dataset for `uv run sft`.

Train and eval runs save episodes as one JSON record per line (prime-rl's FileMonitor
under `rollouts/step_N/<kind>/<subset>/traces.jsonl`, `uv run eval` under
`<run-dir>/traces.jsonl`). An episode holds every agent's trace; only trainable agents'
traces become training data (a judge's trace never does). One row per branch: a linear
rollout contributes one sample, a compacted or branched rollout one per branch — each
branch is a linear root-to-leaf history the trainer can feed.

Rows carry the dataset shape `prime_rl.trainer.sft.data` consumes directly: a `messages`
column (OpenAI chat wire shape) plus a `tools` column (the tools the model was shown,
JSON-encoded — heterogeneous JSON-schema dicts don't fit a fixed Arrow schema).

Selection: generation-errored traces (`stop_condition == "error"`) always drop — a broken
transcript is not a sample. A scoring-only error keeps the generation outcome as its stop
condition and a complete conversation, so it stays; its reward may be partial/zero, which
`--min-reward` handles.

Usage (from the prime-rl repo):
    uv run python tools/sft/vf_to_hf.py <traces.jsonl> --name <dir-or-repo-id>
        [--subset default] [--split train] [--min-reward 1.0] [--drop-truncated] [--push]

Without `--push`, writes `<name>/<subset>/<split>.parquet` and registers it in the
dataset card (`<name>/README.md`), so the trainer loads it with `--data.name <name>
--data.subsets <subset> --data.splits <split>`. Re-running with another subset/split
adds to the same dataset. With `--push`, pushes to the HF Hub repo `<name>` under
config `<subset>` and split `<split>` instead.
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset
from verifiers.v1 import Trace, WireEpisode
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
    if not trace.agent.trainable:
        return False
    if trace.stop_condition == "error":
        return False
    if drop_truncated and trace.is_truncated:
        return False
    return min_reward is None or trace.reward >= min_reward


def register_in_dataset_card(root: Path, subset: str, split: str, rel_path: str) -> None:
    """Point the dataset card's `configs` metadata at the parquet, so
    `load_dataset(root, subset, split=split)` resolves it."""
    readme = root / "README.md"
    meta: dict = {}
    body = ""
    if readme.exists():
        text = readme.read_text()
        if text.startswith("---"):
            _, header, body = text.split("---", 2)
            meta = yaml.safe_load(header) or {}
    configs = meta.setdefault("configs", [])
    config = next((c for c in configs if c["config_name"] == subset), None)
    if config is None:
        config = {"config_name": subset, "data_files": []}
        configs.append(config)
    entry = next((e for e in config["data_files"] if e["split"] == split), None)
    if entry is None:
        config["data_files"].append({"split": split, "path": rel_path})
    else:
        entry["path"] = rel_path
    text = f"---\n{yaml.safe_dump(meta, sort_keys=False)}---{body}"
    readme.write_text(text if text.endswith("\n") else text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, help="a run's traces.jsonl (one episode per line)")
    parser.add_argument("--name", required=True, help="output dataset dir, or HF repo id with --push")
    parser.add_argument("--subset", default="default", help="dataset config name")
    parser.add_argument("--split", default="train", help="dataset split name")
    parser.add_argument("--min-reward", type=float, default=None, help="keep traces with reward >= this")
    parser.add_argument("--drop-truncated", action="store_true", help="drop budget-cut traces")
    parser.add_argument("--push", action="store_true", help="push to the HF Hub instead of writing parquet")
    args = parser.parse_args()

    num_episodes, num_traces, rows = 0, 0, []
    with args.traces.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            num_episodes += 1
            episode = WireEpisode.model_validate(json.loads(line))
            for trace in episode.traces:
                if keep(trace, args.min_reward, args.drop_truncated):
                    num_traces += 1
                    rows.extend(sft_rows(trace))
    print(f"vf-to-hf: {num_episodes} episode(s) -> {num_traces} trainable trace(s) -> {len(rows)} sample(s)")
    if not rows:
        raise SystemExit("vf-to-hf: no samples after selection")

    dataset = Dataset.from_list(rows)
    if args.push:
        dataset.push_to_hub(args.name, config_name=args.subset, split=args.split)
        print(f"vf-to-hf: pushed to {args.name} (subset={args.subset}, split={args.split})")
        return
    root = Path(args.name)
    rel_path = f"{args.subset}/{args.split}.parquet"
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    register_in_dataset_card(root, args.subset, args.split, rel_path)
    print(
        f"vf-to-hf: wrote {path} -> train with "
        f"--data.name {root} --data.subsets {args.subset} --data.splits {args.split}"
    )


if __name__ == "__main__":
    main()
