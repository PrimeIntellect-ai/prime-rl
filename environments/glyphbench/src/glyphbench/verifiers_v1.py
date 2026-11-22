"""Native Verifiers v1 taskset and environment for GlyphBench.

The adapter is intentionally thin: task selection and seeded episode identity
live in the taskset, while the environment drives the public ``BaseGlyphEnv``
reset/step lifecycle.  Prime-RL receives ordinary v1 traces and therefore owns
incremental tokenization, prefix caching, scheduling, and trajectory packing.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator

from glyphbench.core import REGISTRY, make_env
from glyphbench.protocol import (
    build_system_prompt,
    parse_action_response,
    render_user_turn,
)
from glyphbench.termination import is_context_limit_stop

DEFAULT_TASK_ID = "glyphbench/classics-snake-easy-v0"
DEFAULT_SEED = 42
DEFAULT_NUM_SEEDS = 512
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class GlyphBenchTasksetConfig(vf.TasksetConfig):
    """Task selection and rollout protocol shared by training and evaluation."""

    tasks: list[str] = Field(default_factory=list)
    task_file: Path | None = None
    seed: int = DEFAULT_SEED
    num_seeds: int = Field(DEFAULT_NUM_SEEDS, ge=1)
    mixed_seeds: bool = True
    max_output_tokens: int = Field(DEFAULT_MAX_OUTPUT_TOKENS, ge=1)
    forfeit_mode: Literal["freeze", "noop"] = "noop"
    use_memory: Literal[False] = False

    @model_validator(mode="after")
    def _validate_tasks(self):
        if self.task_file is not None and self.tasks:
            raise ValueError("set either tasks or task_file, not both")
        self.resolved_tasks()
        return self

    def resolved_tasks(self) -> list[str]:
        if self.task_file is None:
            tasks = list(self.tasks) or [DEFAULT_TASK_ID]
        else:
            try:
                text = self.task_file.read_text()
            except OSError as exc:
                raise ValueError(f"cannot read task_file {self.task_file}: {exc}") from exc
            tasks = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if not tasks:
            raise ValueError("GlyphBench task selection is empty")
        duplicates = sorted(task for task, count in Counter(tasks).items() if count > 1)
        if duplicates:
            raise ValueError(f"tasks contains duplicates: {duplicates}")
        missing = sorted(set(tasks) - set(REGISTRY))
        if missing:
            raise ValueError(f"unknown GlyphBench task ids: {missing}")
        return tasks


class GlyphBenchTaskData(vf.TaskData):
    task_id: str
    seed: int
    mixed_seeds: bool


class GlyphBenchTask(vf.Task[GlyphBenchTaskData]):
    pass


class GlyphBenchTaskset(vf.Taskset[GlyphBenchTask, GlyphBenchTasksetConfig]):
    """Infinite, balanced stream of task/seed pairs.

    Consecutive examples visit every selected task once before advancing the
    logical seed.  This keeps a single Prime-RL source balanced without one
    environment-server process per task.
    """

    INFINITE = True

    def load(self) -> Iterator[GlyphBenchTask]:
        tasks = self.config.resolved_tasks()
        prompts = {
            task_id: _system_prompt(task_id, self.config)
            for task_id in tasks
        }
        idx = 0
        for seed_offset in itertools.cycle(range(self.config.num_seeds)):
            for task_id in tasks:
                seed = self.config.seed + seed_offset
                yield GlyphBenchTask(
                    GlyphBenchTaskData(
                        idx=idx,
                        name=f"{task_id}#seed={seed}",
                        prompt=None,
                        system_prompt=prompts[task_id],
                        task_id=task_id,
                        seed=seed,
                        mixed_seeds=self.config.mixed_seeds,
                    ),
                    self.config.task,
                )
                idx += 1


class GlyphBenchEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(
        runtime={"type": "subprocess"},
    )


class GlyphBenchEnv(vf.Env[GlyphBenchEnvConfig]):
    """Drive one native GlyphBench episode inside one Verifiers interaction."""

    async def run(self, task: GlyphBenchTask, agents: vf.Agents) -> None:
        config = self.config.taskset
        if not isinstance(config, GlyphBenchTasksetConfig):
            raise TypeError(
                "GlyphBenchEnv requires GlyphBenchTasksetConfig; "
                f"got {type(config).__name__}"
            )

        total_reward = 0.0
        parse_failures = 0
        turns = 0
        terminated = False
        truncated = False
        agent_stopped = False
        last_info: dict = {}
        async with agents.player.interaction(task) as interaction:
            trace = interaction.trace
            seed = rollout_seed(
                task.data.seed,
                trace.id,
                mixed=task.data.mixed_seeds,
            )
            game = make_env(task.data.task_id)
            try:
                current_obs, _ = game.reset(seed)
                while not (terminated or truncated):
                    user_turn = render_user_turn(
                        game,
                        # Conversation history lives once in the Verifiers
                        # message graph. Re-rendering old frames here would
                        # duplicate observations in the causal context.
                        frames=(),
                        current_obs=current_obs,
                        turn=game.turn,
                        max_output_tokens=config.max_output_tokens,
                        enable_reasoning=True,
                    )
                    segment = await interaction.turn(user_turn)
                    if segment.terminated:
                        agent_stopped = True
                        break

                    parsed = parse_action_response(
                        segment.last_reply,
                        game.action_spec,
                        noop=game.noop_action_name,
                    )
                    if parsed.failed:
                        parse_failures += 1
                        if config.forfeit_mode == "freeze":
                            current_obs, reward, terminated, truncated, last_info = (
                                game.forfeit_turn()
                            )
                        else:
                            current_obs, reward, terminated, truncated, last_info = (
                                game.step(parsed.index)
                            )
                    else:
                        current_obs, reward, terminated, truncated, last_info = (
                            game.step(parsed.index)
                        )

                    reward = float(reward)
                    total_reward += reward
                    turns += 1
            finally:
                with suppress(Exception):
                    game.close()

        task_metric = _metric_name(task.data.task_id)
        context_tokens, context_utilization, context_capacity_hit = (
            _context_capacity_stats(trace, self.config.player.max_total_tokens)
        )
        trace.record_reward("episodic_return", total_reward, 1.0)
        trace.record_metrics(
            {
                "episode_length": float(turns),
                "parse_failure_rate": parse_failures / max(turns, 1),
                "terminated": float(terminated),
                "truncated": float(truncated),
                # Framework/agent outcomes are deliberately distinct from
                # native game termination. PrimeRL aggregates these same trace
                # metrics for training and evaluation W&B runs.
                "framework_truncated": float(trace.is_truncated),
                "context_limit_hit": float(
                    is_context_limit_stop(trace.stop_condition)
                ),
                "agent_stopped": float(agent_stopped),
                "context_tokens": float(context_tokens),
                "context_token_utilization": context_utilization,
                "context_capacity_hit": float(context_capacity_hit),
                f"tasks/{task_metric}/episodic_return": total_reward,
            }
        )
        trace.info["glyphbench"] = {
            "task_id": task.data.task_id,
            "configured_seed": task.data.seed,
            "rollout_seed": seed,
            "mixed_seeds": task.data.mixed_seeds,
            "native_max_turns": game.max_turns,
            "turns": turns,
            "terminated": terminated,
            "truncated": truncated,
            "agent_stopped": agent_stopped,
            "context_tokens": context_tokens,
            "context_limit": (
                self.config.player.max_total_tokens + 1
                if self.config.player.max_total_tokens is not None
                else None
            ),
            "context_capacity_hit": context_capacity_hit,
            "trace_stop_condition": trace.stop_condition,
            "last_env_info": _jsonable(last_info),
        }


def rollout_seed(base_seed: int, trace_id: str, *, mixed: bool) -> int:
    """Resolve a replayable rollout seed.

    Prime-RL group members share one task object.  In mixed-seed mode the
    unique trace id supplies the member entropy, and the resolved seed is
    recorded on the trace for exact replay.  Fixed evaluation bypasses this
    mapping and uses the taskset seed directly.
    """

    if not mixed:
        return int(base_seed)
    digest = hashlib.blake2b(
        f"{base_seed}:{trace_id}".encode(),
        digest_size=8,
        person=b"glyphbench-v1",
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _context_capacity_stats(
    trace: vf.Trace,
    max_total_tokens: int | None,
) -> tuple[int, float, bool]:
    """Measure context pressure without conflating it with per-turn length stops."""
    context_tokens = max(
        (branch.num_total_tokens for branch in trace.branches),
        default=0,
    )
    if max_total_tokens is None:
        return context_tokens, 0.0, trace.stop_condition == "context_length"

    context_limit = max_total_tokens + 1
    capacity_hit = (
        trace.stop_condition in {"context_length", "max_total_tokens"}
        or context_tokens >= max_total_tokens
    )
    utilization = min(context_tokens / context_limit, 1.0)
    return context_tokens, utilization, capacity_hit


def _system_prompt(task_id: str, config: GlyphBenchTasksetConfig) -> str:
    game = make_env(task_id)
    try:
        return build_system_prompt(
            game,
            config.max_output_tokens,
            use_memory=False,
            forfeit_mode=config.forfeit_mode,
            enable_reasoning=True,
        )
    finally:
        with suppress(Exception):
            game.close()


def _metric_name(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", task_id).strip("_")


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


__all__ = [
    "GlyphBenchEnv",
    "GlyphBenchEnvConfig",
    "GlyphBenchTask",
    "GlyphBenchTaskData",
    "GlyphBenchTaskset",
    "GlyphBenchTasksetConfig",
    "rollout_seed",
]
