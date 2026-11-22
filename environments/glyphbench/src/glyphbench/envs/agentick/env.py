"""AgenticK task adapter for GlyphBench/Verifiers training."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation

AGENTICK_ACTION_SPEC = ActionSpec(
    names=("NOOP", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "INTERACT"),
    descriptions=(
        "AgenticK action 0: do nothing this turn",
        "AgenticK action 1: move one cell up",
        "AgenticK action 2: move one cell down",
        "AgenticK action 3: move one cell left",
        "AgenticK action 4: move one cell right",
        "AgenticK action 5: interact with the object in the faced cell",
    ),
    extra_aliases={
        "0": "NOOP",
        "1": "MOVE_UP",
        "2": "MOVE_DOWN",
        "3": "MOVE_LEFT",
        "4": "MOVE_RIGHT",
        "5": "INTERACT",
        "UP": "MOVE_UP",
        "DOWN": "MOVE_DOWN",
        "LEFT": "MOVE_LEFT",
        "RIGHT": "MOVE_RIGHT",
    },
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_agentick_importable() -> None:
    try:
        import agentick  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    checkout = _repo_root() / "third_party" / "agentick"
    if (checkout / "agentick" / "__init__.py").exists():
        sys.path.insert(0, str(checkout))
        try:
            import agentick  # noqa: F401

            return
        except ModuleNotFoundError:
            pass

    raise ModuleNotFoundError(
        "AgenticK is required for glyphbench/agentick-* tasks. "
        "Initialize the submodule with "
        "`git submodule update --init --recursive third_party/agentick` "
        "and install it with `uv pip install -e third_party/agentick`."
    )


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text).rstrip("\n")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _compact_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ": "))


def _split_agentick_ascii(raw: Any) -> tuple[str, str, str]:
    text = _strip_ansi(str(raw))
    marker = "\n--- Legend ---\n"
    if marker not in text:
        body = text
        legend = ""
    else:
        body, legend = text.split(marker, 1)

    lines = body.rstrip("\n").splitlines()
    first_grid_line = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            first_grid_line = i
            break
    message = "\n".join(line.rstrip() for line in lines[:first_grid_line]).strip()
    grid_lines = [line.rstrip("\n") for line in lines[first_grid_line:]]
    width = max((len(line) for line in grid_lines), default=0)
    grid = "\n".join(line.ljust(width) for line in grid_lines)
    return grid, legend.strip(), message


class AgentickTaskEnv(BaseGlyphEnv):
    """Hard-difficulty AgenticK task presented through the GlyphBench API."""

    action_spec = AGENTICK_ACTION_SPEC
    noop_action_name = "NOOP"
    default_use_memory = False

    agentick_task_name: str = ""
    task_slug: str = ""
    difficulty: str = "hard"

    def __init__(self, max_turns: int | None = None) -> None:
        self._max_turns_override = max_turns
        super().__init__(max_turns=max_turns or 500)
        self._env: Any | None = None
        self._last_agentick_obs: Any = ""
        self._last_agentick_info: dict[str, Any] = {}
        self._last_message = ""

    def env_id(self) -> str:
        return f"glyphbench/agentick-{self.task_slug}-{self.difficulty}-v0"

    def system_prompt(self) -> str:
        description = self._task_description()
        return (
            "You are playing an AgenticK grid-world task through GlyphBench's "
            "Verifiers training harness.\n\n"
            f"AgenticK task: {self.agentick_task_name}\n"
            f"Difficulty: {self.difficulty}\n\n"
            "This is the AgenticK markovian_reasoner setup: each decision sees "
            "only the current ASCII observation and public HUD/configuration "
            "state. There is no conversation history and no external memory.\n\n"
            "Task objective:\n"
            f"{description}\n\n"
            "AgenticK uses six native action IDs. In this run, emit the "
            "corresponding GlyphBench XML action name instead of the numeric "
            "ID. Sparse reward is +1.0 on task success and 0.0 otherwise; "
            "the episode ends on success or when the step budget is exhausted."
        )

    def _task_description(self) -> str:
        _ensure_agentick_importable()
        from agentick.agents.prompt_templates import get_task_description

        return str(get_task_description(self.agentick_task_name)).strip()

    def _make_agentick_env(self, seed: int) -> Any:
        _ensure_agentick_importable()
        import agentick

        env = agentick.make(
            self.agentick_task_name,
            difficulty=self.difficulty,
            render_mode="ascii",
            reward_mode="sparse",
            seed=seed,
        )
        if self._max_turns_override is not None:
            env.max_steps = int(self._max_turns_override)
        else:
            self.max_turns = int(getattr(env, "max_steps", self.max_turns))
        return env

    def _reset(self, seed: int) -> GridObservation:
        self.close()
        self._env = self._make_agentick_env(seed)
        obs, info = self._env.reset(seed=seed)
        self._last_agentick_obs = obs
        self._last_agentick_info = dict(info)
        if self._max_turns_override is None:
            self.max_turns = int(getattr(self._env, "max_steps", self.max_turns))
        self._last_message = ""
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("call reset() before step()")

        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._last_agentick_obs = obs
        self._last_agentick_info = dict(info)
        action_name = self.action_spec.names[int(action)]
        self._last_message = (
            f"Applied AgenticK action {int(action)} ({action_name}); "
            f"reward={float(reward):.3g}; success={bool(info.get('success', False))}."
        )
        env_info = {
            "agentick_task": self.agentick_task_name,
            "difficulty": self.difficulty,
            "agentick_action_index": int(action),
            "agentick_action_name": self._agentick_action_name(action),
            "success": bool(info.get("success", False)),
            "agentick_info": _jsonable(info),
        }
        return (
            self._render_current_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            env_info,
        )

    def _agentick_action_name(self, action: int) -> str:
        if self._env is None:
            return self.action_spec.names[int(action)]
        return str(self._env.action_space_obj.get_action_name(int(action)))

    def _render_current_observation(self) -> GridObservation:
        if self._env is None:
            raise RuntimeError("call reset() before rendering")
        obs = self._last_agentick_obs
        if not str(obs):
            obs = self._env.render()
        grid, legend, obs_message = _split_agentick_ascii(obs)
        info = dict(self._last_agentick_info)
        if not info:
            info = dict(self._env._get_info())
        public_config = info.get("task_config", {})
        hud_lines = [
            f"AgenticK task: {self.agentick_task_name}",
            f"Difficulty: {self.difficulty}",
            f"Step: {self._turn} / {self.max_turns}",
            f"AgenticK step: {info.get('step_count', info.get('step', '?'))} / "
            f"{info.get('max_steps', self.max_turns)}",
            f"Episode reward: {float(info.get('episode_reward', 0.0)):.3g}",
            f"Agent position: {info.get('agent_position', '?')}",
            "Valid AgenticK actions this turn: "
            + ", ".join(str(a) for a in info.get("valid_actions", [])),
            f"Public task config: {_compact_json(public_config)}",
        ]
        if not legend:
            legend = (
                "^/v/</> or A: agent facing direction\n"
                "#: wall\n"
                ".: floor\n"
                "Task-specific symbols are described in the HUD/config."
            )
        messages = [m for m in (obs_message, self._last_message) if m]
        return GridObservation(
            grid=grid,
            legend=legend,
            hud="\n".join(hud_lines),
            message="\n".join(messages),
        )

    def close(self) -> None:
        env = self._env
        self._env = None
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()


class AgentickInstructionFollowingHardEnv(AgentickTaskEnv):
    agentick_task_name = "InstructionFollowing-v0"
    task_slug = "instruction-following"


class AgentickSokobanPushHardEnv(AgentickTaskEnv):
    agentick_task_name = "SokobanPush-v0"
    task_slug = "sokoban-push"


class AgentickPreciseNavigationHardEnv(AgentickTaskEnv):
    agentick_task_name = "PreciseNavigation-v0"
    task_slug = "precise-navigation"


class AgentickPackingPuzzleHardEnv(AgentickTaskEnv):
    agentick_task_name = "PackingPuzzle-v0"
    task_slug = "packing-puzzle"


class AgentickGraphColoringHardEnv(AgentickTaskEnv):
    agentick_task_name = "GraphColoring-v0"
    task_slug = "graph-coloring"


class AgentickHerdingHardEnv(AgentickTaskEnv):
    agentick_task_name = "Herding-v0"
    task_slug = "herding"


AGENTICK_ENV_CLASSES: tuple[type[AgentickTaskEnv], ...] = (
    AgentickInstructionFollowingHardEnv,
    AgentickSokobanPushHardEnv,
    AgentickPreciseNavigationHardEnv,
    AgentickPackingPuzzleHardEnv,
    AgentickGraphColoringHardEnv,
    AgentickHerdingHardEnv,
)
