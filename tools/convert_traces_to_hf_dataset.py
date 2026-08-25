"""Convert episode traces into a branch-level Hugging Face dataset.

Train and eval runs save one episode per line in `traces.jsonl`. This tool writes one
dataset row for every branch of every agent trace. It does not select or filter rows.

Each row includes `messages` and JSON-encoded `tools` for SFT. It also includes trace,
agent, task, run, outcome, error, timing, and usage metadata. Scalar outcome fields such
as `reward`, `stop_condition`, `has_error`, and `is_truncated` stay as top-level columns
so later scripts can filter them directly. Metadata with variable schemas stays as JSON.

Usage (from the prime-rl repo):
    uv run python tools/convert_traces_to_hf_dataset.py <traces.jsonl> --name <dir-or-repo-id>
        [--subset default] [--split train] [--private] [--local]

By default, the tool pushes to the Hugging Face Hub repo `<name>`. Use `--private` when
creating a private repo. With `--local`, it writes `<name>/<subset>/<split>.parquet` and
registers it in `<name>/README.md` instead.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from pydantic import BaseModel
from verifiers.v1 import Trace, WireEpisode
from verifiers.v1.dialects.chat import message_to_wire


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def as_json(value: Any) -> str:
    return json.dumps(jsonable(value), separators=(",", ":"))


def trace_rows(episode: WireEpisode, trace: Trace) -> list[dict]:
    """Convert each branch of one trace without selecting or filtering it."""
    last_error = trace.last_error
    run = episode.run
    work = getattr(run, "work", None)
    policy = getattr(work, "policy", None)
    common = {
        "episode_id": episode.id,
        "episode_ok": episode.ok,
        "episode_has_error": bool(episode.errors),
        "episode_errors": as_json(episode.errors),
        "env_id": episode.env.id,
        "env_name": episode.env.name,
        "group_id": episode.group.id if episode.group else None,
        "run_type": run.type if run else None,
        "run_id": run.id if run else None,
        "run_name": run.name if run else None,
        "work_type": work.type if work else None,
        "step": getattr(work, "step", None),
        "policy_start": policy.start if policy else None,
        "policy_end": policy.end if policy else None,
        "trace_id": trace.id,
        "trace_version": trace.version,
        "verifiers_version": trace.verifiers.version,
        "verifiers_commit": trace.verifiers.commit,
        "task_type": trace.task.type,
        "task_data": as_json(trace.task.data),
        "task_key": trace.task.key,
        "task_hash": trace.task.hash,
        "agent": trace.agent.name,
        "trainable": trace.agent.trainable,
        "agent_config": as_json(trace.agent.config),
        "agent_runtime": as_json(trace.agent.runtime),
        "tools": as_json(trace.tools),
        "reward": trace.reward,
        "rewards": as_json(trace.rewards),
        "metrics": as_json(trace.metrics),
        "info": as_json(trace.info),
        "is_completed": trace.is_completed,
        "ok": trace.ok,
        "stop_condition": trace.stop_condition,
        "has_error": trace.has_error,
        "error_type": last_error.type if last_error else None,
        "error_message": last_error.message if last_error else None,
        "error_status_code": last_error.status_code if last_error else None,
        "errors": as_json(trace.errors),
        "is_truncated": trace.is_truncated,
        "request_rewrites": as_json(trace.request_rewrites),
        "response_rewrites": as_json(trace.response_rewrites),
        "extra_usage": as_json(trace.extra_usage),
        "timing": as_json(trace.timing),
        "num_branches": trace.num_branches,
        "num_turns": trace.num_turns,
        "num_input_tokens": trace.num_input_tokens,
        "num_output_tokens": trace.num_output_tokens,
        "num_total_tokens": trace.num_total_tokens,
    }
    return [
        {
            **common,
            "branch_index": branch.index,
            "messages": [message_to_wire(m) for m in branch.messages],
            "calls": as_json(branch.calls),
            "branch_num_input_tokens": branch.num_input_tokens,
            "branch_num_output_tokens": branch.num_output_tokens,
            "branch_num_total_tokens": branch.num_total_tokens,
        }
        for branch in trace.branches
    ]


def register_in_dataset_card(root: Path, subset: str, split: str, rel_path: str) -> None:
    """Point the dataset card's `configs` metadata at the parquet, so
    `load_dataset(root, subset, split=split)` resolves it."""
    readme = root / "README.md"
    meta: dict = {}
    body = ""
    if readme.exists():
        text = readme.read_text()
        lines = text.splitlines(keepends=True)
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if lines and lines[0].strip() == "---" and end is not None:
            header = "".join(lines[1:end])
            body = "".join(lines[end + 1 :])
            meta = yaml.safe_load(header) or {}
        else:
            body = text
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
    text = f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n{body}"
    readme.write_text(text if text.endswith("\n") else text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, help="a run's traces.jsonl (one episode per line)")
    parser.add_argument("--name", required=True, help="HF repo id, or output dataset dir with --local")
    parser.add_argument("--subset", default="default", help="dataset config name")
    parser.add_argument("--split", default="train", help="dataset split name")
    parser.add_argument("--private", action="store_true", help="make a new Hub dataset private")
    parser.add_argument("--local", action="store_true", help="write parquet locally instead of pushing")
    args = parser.parse_args()
    if args.local and args.private:
        parser.error("--private cannot be used with --local")

    num_episodes, num_traces, rows = 0, 0, []
    with args.traces.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            num_episodes += 1
            episode = WireEpisode.model_validate(json.loads(line))
            for trace in episode.traces:
                num_traces += 1
                rows.extend(trace_rows(episode, trace))
    print(f"traces-to-hf: {num_episodes} episode(s) -> {num_traces} trace(s) -> {len(rows)} branch(es)")
    if not rows:
        raise SystemExit("traces-to-hf: no branches found")

    dataset = Dataset.from_list(rows)
    if not args.local:
        dataset.push_to_hub(
            args.name,
            config_name=args.subset,
            split=args.split,
            private=True if args.private else None,
        )
        print(f"traces-to-hf: pushed to {args.name} (subset={args.subset}, split={args.split}, private={args.private})")
        return
    root = Path(args.name)
    rel_path = f"{args.subset}/{args.split}.parquet"
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    register_in_dataset_card(root, args.subset, args.split, rel_path)
    print(f"traces-to-hf: wrote {path} (subset={args.subset}, split={args.split})")


if __name__ == "__main__":
    main()
