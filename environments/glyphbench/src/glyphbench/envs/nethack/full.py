"""Full NetHack suite wrapper backed by the local NLE fork."""

from __future__ import annotations

from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.envs.nethack.actions import NETHACK_ACTION_SPEC

NETHACK_NATIVE_MAX_TURNS = 1_000_000


class NetHackFullEnv(BaseGlyphEnv):
    """GlyphBench-compatible adapter for the full NetHack game."""

    action_spec = NETHACK_ACTION_SPEC
    noop_action_name = "WAIT"
    # Open-ended full game: report the RAW unclamped cumulative reward, not the
    # [-1,1] benchmark bound.
    clamp_episode_return = False

    def __init__(
        self,
        max_turns: int = NETHACK_NATIVE_MAX_TURNS,
        *,
        character: str = "@",
        nle_max_episode_steps: int = NETHACK_NATIVE_MAX_TURNS + 1,
    ) -> None:
        super().__init__(max_turns=max_turns)
        self._character = character
        self._nle_max_episode_steps = int(nle_max_episode_steps)
        self._backend: Any | None = None
        self._current: Any | None = None

    def env_id(self) -> str:
        return "glyphbench/nethack-full-v0"

    def _reset(self, seed: int) -> GridObservation:
        backend = self._backend_instance()
        obs, _info = backend.reset(seed)
        self._current = obs
        return self._to_grid_observation(obs)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        backend = self._backend_instance()
        obs, reward, terminated, truncated, info = backend.step(action)
        self._current = obs
        return (
            self._to_grid_observation(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def _render_current_observation(self) -> GridObservation:
        if self._current is None:
            return GridObservation(
                grid="?",
                legend="? — unavailable before reset",
                hud=f"Step: {self.turn} / {self.max_turns}",
                message="Call reset(seed) before reading the NetHack observation.",
            )
        return self._to_grid_observation(self._current)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}, the full NetHack game through "
            "the GlyphBench harness.\n\n"
            "TASK\n"
            "Survive, explore the dungeon, gain score, and ultimately ascend. "
            "This is a held-out long-horizon evaluation suite, not a training "
            "task. Reward is the raw NetHack score delta; the episodic return "
            "reported by the evaluator is the raw score accumulated during the "
            "episode. Death, quitting, or another game-over state does not add "
            "a normalized terminal penalty.\n\n"
            "OBSERVATION\n"
            "The [Grid] is a cropped native NetHack ASCII terminal map centered "
            "on observable non-space characters. Hidden and empty terminal "
            "margins are not shown. Each visible cell is one character; spaces "
            "inside the crop are real NetHack blank/dark cells, so preserve "
            "row/column alignment when reasoning. The [HUD] gives the crop's "
            "terminal row/column origin and the agent's local @ position. "
            "The legend section is generated from NLE screen descriptions for glyphs in "
            "the current crop. The [HUD] also contains bottom-line stats plus "
            "inventory. The [Message] contains NetHack's current message, menu, "
            "or prompt text when present.\n\n"
            "INTERACTION\n"
            "The action list is the full NetHack keyboard command set exposed "
            "as stable GlyphBench action names. Movement uses vi keys in the "
            "backend: MOVE_N/MOVE_E/MOVE_S/MOVE_W and diagonals. GO_DOWN and "
            "GO_UP use stairs. MORE advances --More-- prompts. ESC cancels "
            "menus or prompts. Text/digit actions are available for prompts "
            "that ask for inventory letters, counts, or menu choices.\n\n"
            "MENUS AND PROMPTS\n"
            "NetHack is modal. If the Message asks a question or shows a menu, "
            "choose an action that sends the required key. For yes/no prompts, "
            "MOVE_NW sends the literal 'y' key and MOVE_SE sends the literal "
            "'n' key; ESC cancels many prompts. Inventory letters shown in the "
            "HUD map to text/digit actions only when those keys are listed in "
            "the action set."
        )

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def _backend_instance(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            from nle.glyphbench_wrapper import ACTION_NAMES, GlyphbenchNetHack
        except ImportError as exc:
            raise ImportError(
                "glyphbench/nethack-full-v0 requires the local NLE fork with "
                "the GlyphBench wrapper installed. From this checkout, run "
                "`uv sync --extra nethack` so the `nle` dependency resolves "
                "from `third_party/nle`."
            ) from exc
        if tuple(ACTION_NAMES) != tuple(self.action_spec.names):
            raise RuntimeError(
                "NLE GlyphBench wrapper action names do not match "
                "glyphbench.envs.nethack.actions.NETHACK_ACTION_SPEC."
            )
        self._backend = GlyphbenchNetHack(
            max_episode_steps=max(
                self._nle_max_episode_steps,
                int(self.max_turns) + 1,
            ),
            character=self._character,
        )
        return self._backend

    @staticmethod
    def _to_grid_observation(obs: Any) -> GridObservation:
        return GridObservation(
            grid=str(obs.grid),
            legend=str(obs.legend),
            hud=str(obs.hud),
            message=str(obs.message),
        )
