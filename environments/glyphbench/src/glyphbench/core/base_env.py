"""Base class for every game in glyphbench.

Plain Python class — no framework inheritance. Subclasses implement the five
abstract methods; the public reset/step surface is fixed here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation


class BaseGlyphEnv(ABC):
    """Base class for every env in glyphbench.

    Subclasses MUST:
      - set `action_spec: ActionSpec` as a class or instance attribute
      - implement `_reset(seed)` returning the initial GridObservation
      - implement `_step(action_index)` returning (obs, reward, terminated, truncated, info)
      - implement `_render_current_observation()` returning current state as GridObservation
      - implement `system_prompt()` returning the per-game system prompt
      - implement `env_id()` returning the canonical env id string
    """

    action_spec: ActionSpec
    noop_action_name: str = "NOOP"
    # Whether the verifiers integration should enable the two-call memory
    # scaffold when callers ask for a task's default. Fully observable,
    # Markovian tasks should leave this false; partially observable or
    # hidden-cue tasks opt in through the registry metadata.
    default_use_memory: bool = False
    # Whether step() clamps each reward so the cumulative episode return stays
    # within [-1, 1]. True for the scored benchmark set (bounded-return
    # invariant). The open-ended "full" games (craftaxfull / nethack-full) set
    # this False so they report the env's RAW unclamped cumulative reward.
    clamp_episode_return: bool = True

    def __init__(self, max_turns: int = 500) -> None:
        self.max_turns = max_turns
        self._turn: int = 0
        self._rng: np.random.Generator | None = None
        self._episode_done: bool = False
        self._episode_terminated: bool = False
        self._episode_truncated: bool = False
        self._last_observation: GridObservation | None = None
        # Running cumulative reward, used to enforce the [-1, 1] return bound
        # structurally for bounded tasks (see step()).
        self._episode_return: float = 0.0

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self._rng = np.random.default_rng(int(seed))
        self._turn = 0
        self._episode_return = 0.0
        self._episode_done = False
        self._episode_terminated = False
        self._episode_truncated = False
        obs = self._reset(int(seed))
        self._last_observation = obs
        info: dict[str, Any] = {
            "turn": 0,
            "env_id": self.env_id(),
            "seed": int(seed),
        }
        return obs.render(), info

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= int(action) < self.action_spec.n:
            raise ValueError(
                f"action {action} out of range [0, {self.action_spec.n})"
            )
        if self._episode_done:
            obs = self._last_observation or self._render_current_observation()
            info: dict[str, Any] = {
                "turn": self._turn,
                "env_id": self.env_id(),
                "already_done": True,
            }
            return (
                obs.render(),
                0.0,
                self._episode_terminated,
                self._episode_truncated,
                info,
            )
        self._turn += 1
        obs, reward, terminated, truncated, info = self._step(int(action))
        if self._turn >= self.max_turns and not (terminated or truncated):
            truncated = True
            info["truncation_reason"] = "max_turns"
        info["turn"] = self._turn
        info["env_id"] = self.env_id()
        self._last_observation = obs
        # Enforce the [-1, 1] cumulative-return invariant structurally for the
        # scored benchmark set: clamp this step's reward so the running
        # episode return cannot leave [-1, 1]. For correctly-bounded envs this
        # never engages; it is a safety net that makes the bound a framework
        # guarantee rather than per-env discipline. The open-ended "full" games
        # (craftaxfull / nethack-full) opt out via clamp_episode_return=False so
        # they expose the env's raw unclamped cumulative reward.
        reward = float(reward)
        if self.clamp_episode_return:
            lo = -1.0 - self._episode_return
            hi = 1.0 - self._episode_return
            if reward < lo:
                reward = lo
            elif reward > hi:
                reward = hi
        self._episode_return += reward
        if terminated or truncated:
            self._episode_done = True
            self._episode_terminated = terminated
            self._episode_truncated = truncated
        return obs.render(), reward, terminated, truncated, info

    def forfeit_turn(self) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Advance the turn counter without stepping the env.

        Used when the LLM's action could not be parsed: the env state is
        unchanged, observation is re-rendered with the bumped Step counter
        in the HUD, reward is 0, and truncation fires only if the bump hit
        ``max_turns``.

        Returns the same 5-tuple shape as ``step``.
        """
        if self._episode_done:
            obs = self._last_observation or self._render_current_observation()
            return (
                obs.render(),
                0.0,
                self._episode_terminated,
                self._episode_truncated,
                {
                    "turn": self._turn,
                    "env_id": self.env_id(),
                    "forfeit": True,
                    "already_done": True,
                },
            )
        self._turn += 1
        truncated = False
        info: dict[str, Any] = {
            "turn": self._turn,
            "env_id": self.env_id(),
            "forfeit": True,
        }
        if self._turn >= self.max_turns:
            truncated = True
            info["truncation_reason"] = "max_turns"
        obs = self._render_current_observation()
        self._last_observation = obs
        if truncated:
            self._episode_done = True
            self._episode_terminated = False
            self._episode_truncated = True
        return obs.render(), 0.0, False, truncated, info

    @abstractmethod
    def _reset(self, seed: int) -> GridObservation: ...

    @abstractmethod
    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]: ...

    @abstractmethod
    def _render_current_observation(self) -> GridObservation: ...

    @abstractmethod
    def system_prompt(self) -> str:
        """Per-game rules / goal description for the system prompt.

        Should describe the objective, reward structure, and any game-specific
        mechanics. The action list is optional: ``build_system_prompt`` in
        ``verifiers_integration.prompting`` appends
        ``action_spec.render_for_prompt()`` when it is not already present.
        """
        ...

    @abstractmethod
    def env_id(self) -> str: ...

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() before accessing rng")
        return self._rng

    @property
    def turn(self) -> int:
        """Number of steps taken since the last reset()."""
        return self._turn

    def get_observation(self) -> GridObservation:
        """Return the current observation without stepping. Useful for initial
        prompt construction at turn 0."""
        return self._render_current_observation()

    def close(self) -> None:
        """Optional cleanup hook. Default: no-op."""
        return None
