"""Trace to SFT rows: stored episodes as the ``{messages, tools}`` rows prime-rl's SFT
trainer reads (``trainer/sft/data.py``).

A row is one trainable branch of one trace (``vf.Trace.branches``: a root-to-leaf path
of its message graph, a compaction opening a new one), the same unit RL trains on:

    {"messages": [...], "tools": [...], "meta": {episode, trace, branch, env, task,
                                                 agent, reward, labels}}

``messages`` are each message as the wire carried it (``message_to_wire``: tool-call
arguments a string, a tool result under its ``tool_call_id``), with text-only content
as one string and the reasoning as ``reasoning_content`` (a provider's opaque
``reasoning_details`` is not text to train on). ``tools`` are the branch's tools in the
OpenAI ``{"type": "function", "function": {...}}`` envelope every renderer reads. An
image's data URL is written once per content hash beside the rows when ``Images`` is
given, and its part points at that file.

Which episodes make good rows is the caller's: ``export`` keeps ok episodes and can
filter by kind, env, agent and reward; a pipeline with its own selection or windowing
uses ``sft_message`` and ``sft_tools`` on its own messages.

    uv run python -m prime_rl.monitors.file.traces.sft <run_dir>... --out sft.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson
import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire

from prime_rl.eval.resume import archives, read_records
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.index import episode_kind
from prime_rl.utils.pathing import get_file_monitor_dir

IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp", "image/gif": "gif"}


class Images:
    """Image data URLs written once each, by content, as ``<sha256>.<ext>`` under
    ``directory``; ``file`` answers the path relative to ``relative_to`` (the rows file's
    directory) that an ``image_url`` part then holds. A URL that is not a base64 data URL
    of a known image type (an address the model was given) is kept as it was."""

    def __init__(self, directory: Path, relative_to: Path) -> None:
        self.directory = directory
        self.relative_to = relative_to
        self.written: dict[str, str] = {}

    def file(self, url: str) -> str:
        if not url.startswith("data:"):
            return url
        header, _, payload = url.removeprefix("data:").partition(",")
        ext = IMAGE_TYPES.get(header.split(";", 1)[0])
        if ext is None or not header.endswith(";base64"):
            return url
        data = base64.b64decode(payload)
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self.written:
            path = self.directory / f"{sha}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            self.written[sha] = os.path.relpath(path, self.relative_to)
        return self.written[sha]


def sft_message(message: vf.Message, images: Images | None = None) -> dict[str, Any]:
    """One message as an SFT row carries it (see the module docstring)."""
    wire = message_to_wire(message)
    if wire.pop("reasoning_details", None) is not None and getattr(message, "reasoning_content", None):
        wire["reasoning_content"] = message.reasoning_content
    content = wire.get("content")
    if isinstance(content, list):
        if all(part.get("type") == "text" for part in content):
            wire["content"] = "".join(part.get("text") or "" for part in content)
        elif images is not None:
            for part in content:
                if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                    part["image_url"]["url"] = images.file(part["image_url"]["url"])
    return wire


def sft_tools(tools: Iterable[vf.Tool]) -> list[dict[str, Any]]:
    """Tools in the OpenAI function envelope; unset optional fields left out."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                **({"strict": tool.strict} if tool.strict is not None else {}),
            },
        }
        for tool in tools
    ]


def trace_rows(episode: vf.WireEpisode, trace: vf.Trace, images: Images | None = None) -> Iterator[dict[str, Any]]:
    """One row per trainable branch of ``trace`` (a compaction attempt the harness did
    not resume from is not one), oldest branch first; a branch with no assistant
    message has nothing to learn and is skipped."""
    for branch in trace.branches:
        if not branch.trainable or not any(message.role == "assistant" for message in branch.messages):
            continue
        yield {
            "messages": [sft_message(message, images) for message in branch.messages],
            "tools": sft_tools(branch.tools),
            "meta": {
                "episode": episode.id,
                "trace": trace.id,
                "branch": branch.index,
                "env": episode.env.name or episode.env.id,
                "task": episode.task.key,
                "agent": trace.agent.name,
                "reward": trace.reward,
                "labels": trace.labels,
            },
        }


def run_records(run_dir: Path) -> Iterator[dict]:
    """Every episode record of a run, each episode once (its last record wins), from the
    stream of every attempt: earlier ones (``monitors/file.attempt_N``) then the current."""
    current = get_file_monitor_dir(run_dir)
    stream = get_trace_stream(run_dir).relative_to(current)
    records: dict[str, dict] = {}
    for directory in [*archives(run_dir), current]:
        if (directory / stream).is_dir():
            for record in read_records(directory / stream):
                if isinstance(record, dict) and "traces" in record:
                    records.pop(record.get("id"), None)
                    records[record.get("id")] = record
    yield from records.values()


def export(
    run_dirs: Iterable[Path],
    dest: Path,
    *,
    include_errored: bool = False,
    kinds: Iterable[str] = (),
    envs: Iterable[str] = (),
    agents: Iterable[str] = (),
    min_reward: float | None = None,
    inline_images: bool = False,
) -> int:
    """Write the SFT rows of every episode of ``run_dirs`` to ``dest``, replacing it
    whole (temp file + rename), and return the row count. Kept: ok episodes (unless
    ``include_errored``), of the given ``kinds`` (``train``/``eval``), ``envs`` and
    ``agents`` when any are named, and traces whose reward is at least ``min_reward``.
    Images go to ``<dest stem>-images/`` beside ``dest`` unless ``inline_images`` keeps
    their data URLs."""
    kinds, envs, agents = set(kinds), set(envs), set(agents)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    images = None if inline_images else Images(dest.with_name(f"{dest.stem}-images"), dest.parent)
    written = 0
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.")
    try:
        with os.fdopen(fd, "wb") as out:
            for run_dir in run_dirs:
                for record in run_records(Path(run_dir)):
                    if not (include_errored or record.get("ok")):
                        continue
                    if kinds and episode_kind(record) not in kinds:
                        continue
                    episode = vf.WireEpisode.model_validate(record)
                    if envs and (episode.env.name or episode.env.id) not in envs:
                        continue
                    for trace in episode.traces:
                        if agents and trace.agent.name not in agents:
                            continue
                        if min_reward is not None and trace.reward < min_reward:
                            continue
                        for row in trace_rows(episode, trace, images):
                            out.write(orjson.dumps(row) + b"\n")
                            written += 1
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp's 0600 is for the temp file, not the rows
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="prime-rl run output directories")
    parser.add_argument("--out", type=Path, required=True, help="the JSONL file to write")
    parser.add_argument("--include-errored", action="store_true", help="also export episodes that are not ok")
    parser.add_argument("--kind", action="append", default=[], help="train or eval (repeatable)")
    parser.add_argument("--env", action="append", default=[], help="an env name (repeatable)")
    parser.add_argument("--agent", action="append", default=[], help="an agent name (repeatable)")
    parser.add_argument("--min-reward", type=float, default=None, help="keep traces scoring at least this")
    parser.add_argument("--inline-images", action="store_true", help="keep image data URLs in the rows")
    args = parser.parse_args(argv)
    count = export(
        args.run_dirs,
        args.out,
        include_errored=args.include_errored,
        kinds=args.kind,
        envs=args.env,
        agents=args.agent,
        min_reward=args.min_reward,
        inline_images=args.inline_images,
    )
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
