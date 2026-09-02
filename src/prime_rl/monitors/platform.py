"""The Prime platform's sample format and credentials, shared by the train and eval
monitors: one sample per v1 ``Episode``, with the complete native episode as its source of
truth and a flat summary for older platform consumers."""

from __future__ import annotations

import json
import os
from typing import Any

import verifiers.v1 as vf
from prime_cli.core.config import Config as PrimeConfig

from prime_rl.utils.logger import get_logger

API_KEY_VAR = "PRIME_API_KEY"

# Repeated /samples posts append; match the platform's request ceiling.
MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024
EMPTY_SAMPLES_PAYLOAD_BYTES = len(b'{"samples":[]}')


def credentials() -> tuple[str | None, str, str, str | None]:
    """``(api_key, api_base, frontend_url, team_id)``: environment variables first, then
    the prime CLI config (``prime login``)."""
    config = PrimeConfig()
    api_key = os.getenv(API_KEY_VAR) or config.api_key
    api_base = (os.getenv("PRIME_API_BASE_URL") or config.base_url).rstrip("/").removesuffix("/api/v1")
    frontend_url = os.getenv("PRIME_FRONTEND_URL") or config.frontend_url
    team_id = os.getenv("PRIME_TEAM_ID") or config.team_id
    return api_key, api_base, frontend_url, team_id


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def trace_to_sample(trace: vf.Trace, rollout_number: int = 1, episode_id: str | None = None) -> dict[str, Any]:
    """One trace -> the platform's sample dict (the v0 eval-sample format).

    The table stays flat — one row per trace; its episode is denormalized onto the row
    (``episode_id`` from the envelope, plus the trace's own ``agent``/``trainable``), so a
    multi-trace rollout's grouping travels with each row without a nested schema. No
    prompt/completion split (meaningless mid-branch): ``completion`` is the final branch's
    messages, ``trajectory`` one message list per branch."""

    def dump(messages):
        return [m.model_dump(mode="json", exclude_none=True) for m in messages]

    task = trace.task.data.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    sample = {
        "sample_id": trace.id,
        "example_id": trace.task.data.idx,
        "rollout_number": rollout_number,
        "episode_id": episode_id,
        "agent": trace.agent.name,
        "trainable": trace.agent.trainable,
        "task": task,
        "prompt": [],
        "completion": dump(branches[-1].messages) if branches else [],
        "answer": task.get("answer"),
        # Keyed ``tool_defs`` because the v0 sample format already carries it there.
        "tool_defs": [t.model_dump(mode="json", exclude_none=True) for t in trace.tools] if trace.tools else None,
        "reward": trace.reward,
        "timing": trace.timing.model_dump(mode="json", exclude_none=True),
        "is_completed": trace.is_completed,
        "is_truncated": trace.is_truncated,
        "metrics": trace.metrics,
        "error": trace.last_error.model_dump(mode="json", exclude_none=True) if trace.last_error else None,
        "stop_condition": trace.stop_condition,
        "trajectory": [
            {
                "messages": dump(branch.messages),
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "token_usage": trace.usage.model_dump(mode="json", exclude_none=True) if trace.usage else None,
        "info": dict(trace.info) or None,
    }
    # Flatten sub-rewards to top-level keys the way v0 does (raw scores); env metrics stay nested.
    for name, reward in trace.rewards.items():
        if reward is not None:
            sample.setdefault(name, reward.score)
    return sample


def run_metrics(episodes: list[vf.Episode], traces: list[vf.Trace]) -> dict[str, Any]:
    """Run-level aggregates. Rewards/metrics aggregate over the trainable traces only —
    fixed agents (a judge, a modeled user) often carry no rewards and would dilute every
    mean with structural zeros — falling back to all traces when none are trainable.
    ``avg_error`` is the share of episodes that aren't ok: a hook failure counts even when
    its traces are clean or it left none."""
    scored = [t for t in traces if t.agent.trainable] or traces
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for trace in scored:
        scores = {name: reward.score for name, reward in trace.rewards.items() if reward is not None}
        metrics = {name: value for name, value in trace.metrics.items() if value is not None}
        for name, value in {**scores, **metrics}.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    n = len(scored)
    avg_error = sum(not e.ok for e in episodes) / len(episodes) if episodes else 0.0
    return {
        "avg_reward": sum(t.reward for t in scored) / n if n else 0.0,
        "avg_metrics": {name: sums[name] / counts[name] for name in sums},
        "avg_error": avg_error,
    }


def build_samples(episodes: list[vf.Episode]) -> list[dict[str, Any]]:
    """One platform sample per episode, with a legacy-compatible trace summary.

    The native episode in ``info.native_wrapper`` is authoritative and contains every
    trace. One trainable trace (or the first trace) supplies only the flat summary used by
    older consumers; ``native_trace_index`` identifies that summary trace. An episode too
    large for one request falls back to one projected sample per trace."""
    counts: dict[int, int] = {}
    samples = []
    for episode in episodes:
        if not episode.traces:
            continue
        summary_trace_index = next(
            (index for index, candidate in enumerate(episode.traces) if candidate.agent.trainable), 0
        )
        summary_trace = episode.traces[summary_trace_index]
        idx = summary_trace.task.data.idx
        counts[idx] = number = counts.get(idx, 0) + 1
        sample = trace_to_sample(summary_trace, number, episode.id)
        sample["sample_id"] = episode.id
        sample["info"] = {
            **(sample["info"] or {}),
            "native_wrapper": episode.to_record(),
            "native_trace_index": summary_trace_index,
        }
        if EMPTY_SAMPLES_PAYLOAD_BYTES + json_bytes(sample) <= MAX_SAMPLES_PAYLOAD_BYTES:
            samples.append(sample)
            continue
        get_logger().warning(f"Episode {episode.id} exceeds the platform sample limit; uploading projected traces")
        samples.extend(trace_to_sample(candidate, number, episode.id) for candidate in episode.traces)
    return samples
