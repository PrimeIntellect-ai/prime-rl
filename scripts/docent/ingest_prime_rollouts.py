#!/usr/bin/env python3
"""Ingest Prime RL rollout JSONL snapshots into a Docent collection.

Run with:
    uv run --no-project --with docent-python python scripts/docent/ingest_prime_rollouts.py ...
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

FULL_ROLLOUT_GLOBS = ("train_rollouts_full.jsonl", "eval_rollouts_*.full.jsonl")
REDUCED_ROLLOUT_GLOB = "*rollouts*.jsonl"


def _rollout_kind(path: Path) -> str:
    name = path.name
    if name.startswith("eval_rollouts"):
        return "eval"
    if name.startswith("train_rollouts"):
        return "train"
    return "unknown"


def _step_from_path(path: Path) -> int | None:
    for parent in [path.parent, *path.parents]:
        if parent.name.startswith("step_"):
            suffix = parent.name.removeprefix("step_")
            return int(suffix) if suffix.isdigit() else None
    return None


def iter_rollout_files(paths: Iterable[Path], *, include_reduced: bool) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            patterns = list(FULL_ROLLOUT_GLOBS)
            if include_reduced:
                patterns.append(REDUCED_ROLLOUT_GLOB)
            for pattern in patterns:
                for match in sorted(path.rglob(pattern)):
                    if match in seen:
                        continue
                    seen.add(match)
                    yield match
            continue

        if path in seen:
            continue
        seen.add(path)
        yield path


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def _as_message_list(value: Any, default_role: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    if isinstance(value, dict):
        if "role" in value:
            return [value]
        return [{"role": default_role, "content": json.dumps(value, sort_keys=True)}]
    if isinstance(value, list):
        if all(isinstance(item, dict) and "role" in item for item in value):
            return value
        return [{"role": default_role, "content": json.dumps(value, sort_keys=True)}]
    return [{"role": default_role, "content": str(value)}]


def _json_arguments(raw_arguments: Any) -> Any:
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw": raw_arguments}


def _tool_call_dict(raw_tool_call: Any) -> dict[str, Any]:
    if isinstance(raw_tool_call, dict):
        return raw_tool_call
    if isinstance(raw_tool_call, str):
        try:
            parsed = json.loads(raw_tool_call)
        except json.JSONDecodeError:
            return {"function": {"name": "unknown", "arguments": {"raw": raw_tool_call}}}
        if isinstance(parsed, dict):
            return parsed
    return {"function": {"name": "unknown", "arguments": {"raw": raw_tool_call}}}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, default=str)


def _normalize_message(raw: dict[str, Any], *, step_index: int | None) -> dict[str, Any]:
    role = raw.get("role")
    normalized_role = "system" if role == "developer" else role
    if normalized_role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported chat role {role!r}")

    metadata = dict(raw.get("metadata") or {})
    if role != normalized_role:
        metadata["prime_rl_original_role"] = role
    if step_index is not None:
        metadata["prime_rl_step_index"] = step_index

    message: dict[str, Any] = {
        "role": normalized_role,
        "content": _content_text(raw.get("content")),
    }
    if metadata:
        message["metadata"] = metadata

    if normalized_role == "tool":
        if raw.get("name") is not None:
            message["function"] = raw["name"]
        if raw.get("tool_call_id") is not None:
            message["tool_call_id"] = raw["tool_call_id"]

    if normalized_role == "assistant" and raw.get("tool_calls"):
        from docent.data_models.chat import ToolCall

        tool_calls = []
        for tool_call_index, raw_tool_call in enumerate(raw["tool_calls"]):
            tool_call = _tool_call_dict(raw_tool_call)
            raw_function = tool_call.get("function") or {}
            function = raw_function if isinstance(raw_function, dict) else {"name": str(raw_function)}
            function_name = function.get("name") or tool_call.get("name") or "unknown"
            tool_calls.append(
                ToolCall(
                    id=tool_call.get("id") or f"tool_call_{step_index}_{tool_call_index}",
                    function=function_name,
                    arguments=_json_arguments(function.get("arguments", {})),
                    type=tool_call.get("type", "function"),
                    parse_error=None,
                )
            )
        message["tool_calls"] = tool_calls

    return message


def _messages_from_prompt_completion(
    prompt: Any,
    completion: Any,
    *,
    step_index: int | None,
) -> list[dict[str, Any]]:
    raw_messages = [
        *_as_message_list(prompt, "user"),
        *_as_message_list(completion, "assistant"),
    ]
    return [_normalize_message(message, step_index=step_index) for message in raw_messages]


def _score_metadata(rollout: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for key in ("reward", "advantage"):
        if key in rollout and rollout[key] is not None:
            scores[key] = rollout[key]
    if rollout.get("metrics"):
        scores["metrics"] = rollout["metrics"]
    return scores


def _agent_run_metadata(
    rollout: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    source_label: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": "prime-rl",
        "source_file": str(path),
        "source_line": line_number,
        "source_step": _step_from_path(path),
        "rollout_kind": _rollout_kind(path),
        "source_label": source_label,
        "scores": _score_metadata(rollout),
    }
    for key in (
        "rollout_id",
        "group_id",
        "env_name",
        "example_id",
        "policy_version",
        "off_policy_steps",
        "eval_step",
        "task",
        "answer",
        "is_truncated",
        "error",
        "filters",
        "timing",
        "token_usage",
    ):
        if key in rollout and rollout[key] is not None:
            metadata[key] = rollout[key]
    return {key: value for key, value in metadata.items() if value is not None}


def build_agent_run(
    rollout: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    allow_reduced: bool,
    source_label: str | None,
):
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message

    transcripts = []
    trajectory = rollout.get("trajectory") or []
    for step_index, step in enumerate(trajectory):
        messages = _messages_from_prompt_completion(
            step.get("prompt"),
            step.get("completion"),
            step_index=step_index,
        )
        if not messages:
            continue
        transcripts.append(
            Transcript(
                name=f"trajectory_step_{step_index}",
                messages=[parse_chat_message(message) for message in messages],
                metadata={
                    "step_index": step_index,
                    "reward": step.get("reward"),
                    "advantage": step.get("advantage"),
                    "extras": step.get("extras") or {},
                    "has_tokens": step.get("tokens") is not None,
                },
            )
        )

    if not transcripts and allow_reduced:
        messages = _messages_from_prompt_completion(
            rollout.get("prompt"),
            rollout.get("completion"),
            step_index=None,
        )
        if messages:
            transcripts.append(
                Transcript(
                    name="reduced_rollout",
                    messages=[parse_chat_message(message) for message in messages],
                    metadata={"reduced": True},
                )
            )

    if not transcripts:
        raise ValueError(
            f"{path}:{line_number} has no trajectory. Re-run with save_full_rollouts=true "
            "or pass --allow-reduced to ingest top-level prompt/completion only."
        )

    metadata = _agent_run_metadata(
        rollout,
        path=path,
        line_number=line_number,
        source_label=source_label,
    )
    name_parts = [
        str(metadata.get("env_name", "unknown-env")),
        str(metadata.get("example_id", "unknown-example")),
        str(metadata.get("rollout_id", line_number)),
    ]
    return AgentRun(
        name=":".join(name_parts),
        transcripts=transcripts,
        metadata=metadata,
    )


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Rollout JSONL files or directories containing them.")
    collection = parser.add_mutually_exclusive_group(required=False)
    collection.add_argument("--collection-id", help="Existing Docent collection ID.")
    collection.add_argument("--collection-name", help="Create a new Docent collection with this name.")
    parser.add_argument(
        "--description", default="Prime RL transcript analysis", help="Description for a new collection."
    )
    parser.add_argument("--source-label", help="Optional label stored on each ingested AgentRun.")
    parser.add_argument("--batch-size", type=int, default=100, help="AgentRun upload batch size.")
    parser.add_argument("--limit", type=int, help="Maximum rollout rows to ingest.")
    parser.add_argument(
        "--allow-reduced",
        action="store_true",
        help="Allow reduced rollout JSONL without trajectory by ingesting top-level prompt/completion only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs and print counts without uploading.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.dry_run and not (args.collection_id or args.collection_name):
        parser.error("one of --collection-id or --collection-name is required unless --dry-run is set")
    return args


def main() -> None:
    args = parse_args()
    paths = list(iter_rollout_files(args.paths, include_reduced=args.allow_reduced))
    if not paths:
        raise FileNotFoundError("No rollout JSONL files found")

    agent_runs = []
    for path in paths:
        for line_number, rollout in iter_jsonl(path):
            agent_runs.append(
                build_agent_run(
                    rollout,
                    path=path,
                    line_number=line_number,
                    allow_reduced=args.allow_reduced,
                    source_label=args.source_label,
                )
            )
            if args.limit is not None and len(agent_runs) >= args.limit:
                break
        if args.limit is not None and len(agent_runs) >= args.limit:
            break

    print(f"Prepared {len(agent_runs)} AgentRun(s) from {len(paths)} file(s).")
    if args.dry_run:
        return

    from docent import Docent

    client = Docent()
    collection_id = args.collection_id
    if collection_id is None:
        collection_id = client.create_collection(
            name=args.collection_name,
            description=args.description,
            metadata={"source": "prime-rl", "source_label": args.source_label},
        )
        print(f"Created collection: {collection_id}")

    for batch in _chunks(agent_runs, args.batch_size):
        client.add_agent_runs(collection_id, batch)
        print(f"Uploaded {len(batch)} AgentRun(s).")

    print(f"Done. Collection ID: {collection_id}")


if __name__ == "__main__":
    main()
