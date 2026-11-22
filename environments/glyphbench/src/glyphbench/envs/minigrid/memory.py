"""MiniGrid Memory environments.

Long corridor with one or two disappearing cue objects near the start and a
matching-object choice at the far fork. Per the spoiler convention, the cue
answer is *never* leaked through the system prompt — the cue must be observed
on the rendered grid before vanishing.

Variants:
- s7  : type-recall  — one cue (Ball or Key); fork has both types in the SAME
        colour, agent must pick the matching type. Colour is irrelevant.
- s9  : colour-recall — one cue with a colour; fork has TWO objects of the SAME
        type but different colours; agent must pick the matching colour.
- s11 : colour-recall (larger corridor; otherwise identical to s9).
- s13 : bind-and-recall — TWO cues (a Ball of one colour and a Key of another).
        The task description names which TYPE to pick at the fork; the agent
        must remember which COLOUR was on that type. Fork has two same-type
        objects of two different colours.
- s17 : bind-and-recall (larger corridor; otherwise identical to s13).
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Key, Wall

_COLORS = ["red", "green", "blue", "yellow", "purple"]
_OBJ_TYPES = [Ball, Key]


# ---------------------------------------------------------------------------
# Shared base — handles corridor layout, cue erasure, fork termination.
# ---------------------------------------------------------------------------


class _MemoryBase(MiniGridBase):
    _grid_size: int = 7
    # Subclass mode flag; one of "type", "color", "bind".
    _mode: str = "type"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        # Single-cue state (used by type / color modes).
        self._memory_object_desc: str = ""
        self._cue_color: str = ""
        self._cue_type: type = Ball
        self._cue_pos: tuple[int, int] = (0, 0)
        # Two-cue state (used by bind mode).
        self._cue_pos_a: tuple[int, int] = (0, 0)
        self._cue_pos_b: tuple[int, int] = (0, 0)
        # Which TYPE is the fork-question target (bind mode only).
        self._target_type: type = Ball
        self._target_type_name: str = "ball"
        # Targets at the fork.
        self._correct_target_pos: tuple[int, int] = (0, 0)
        self._wrong_target_pos: tuple[int, int] = (0, 0)
        self._cue_visible: bool = False
        self._fork_visible: bool = False
        self._correct_target_obj: Any | None = None
        self._wrong_target_obj: Any | None = None

    # -- corridor layout -------------------------------------------------

    def _carve_corridor(self) -> tuple[int, int, int]:
        s = self._grid_size
        self._init_grid(s, s)
        corridor_y = s // 2
        branch_x = s - 3
        target_x = s - 2

        for y in range(1, s - 1):
            for x in range(1, s - 1):
                self._place_obj(x, y, Wall())
        for x in range(1, branch_x + 1):
            self._grid[corridor_y][x] = None
        self._grid[corridor_y - 1][branch_x] = None
        self._grid[corridor_y + 1][branch_x] = None
        return corridor_y, branch_x, target_x

    # -- mode-specific generation ---------------------------------------

    def _gen_type_recall(self, corridor_y: int, target_x: int) -> None:
        """One cue; fork has both types in the SAME (random) colour."""
        cue_type = _OBJ_TYPES[int(self.rng.integers(0, len(_OBJ_TYPES)))]
        # Both fork objects share this colour — colour conveys no info.
        fork_color = _COLORS[int(self.rng.integers(0, len(_COLORS)))]
        cue_color = _COLORS[int(self.rng.integers(0, len(_COLORS)))]
        cue_obj = cue_type(color=cue_color)
        self._cue_color = cue_color
        self._cue_type = cue_type
        self._memory_object_desc = cue_obj.legend_name()
        self._cue_pos = (2, max(1, corridor_y - 2))
        self._place_obj(*self._cue_pos, cue_obj)
        self._cue_visible = True

        # The "correct" object matches the cue TYPE; the wrong one is the
        # other type — both with the same fork_color (so type is the only signal).
        other_type = Key if cue_type is Ball else Ball
        correct_obj = cue_type(color=fork_color)
        wrong_obj = other_type(color=fork_color)
        self._place_fork_objects(corridor_y, target_x, correct_obj, wrong_obj)

    def _gen_color_recall(self, corridor_y: int, target_x: int) -> None:
        """One cue; fork has SAME type, two different colours."""
        cue_type = _OBJ_TYPES[int(self.rng.integers(0, len(_OBJ_TYPES)))]
        cue_color = _COLORS[int(self.rng.integers(0, len(_COLORS)))]
        # Avoid collision: shift past cue_color if equal.
        idx_d = _COLORS.index(cue_color)
        candidates = [c for i, c in enumerate(_COLORS) if i != idx_d]
        distractor_color = candidates[int(self.rng.integers(0, len(candidates)))]

        cue_obj = cue_type(color=cue_color)
        self._cue_color = cue_color
        self._cue_type = cue_type
        self._memory_object_desc = cue_obj.legend_name()
        self._cue_pos = (2, max(1, corridor_y - 2))
        self._place_obj(*self._cue_pos, cue_obj)
        self._cue_visible = True

        correct_obj = cue_type(color=cue_color)
        wrong_obj = cue_type(color=distractor_color)
        self._place_fork_objects(corridor_y, target_x, correct_obj, wrong_obj)

    def _gen_bind_and_recall(self, corridor_y: int, target_x: int) -> None:
        """Two cues — a Ball of one colour AND a Key of another.

        The task description tells the agent WHICH TYPE the fork question
        targets (e.g. "...pick up the ball at the fork"). The agent must
        remember which COLOUR appeared on that type during the cue phase.
        Fork has two objects of the chosen TYPE in two different colours.
        """
        # Two distinct colours for the two cue objects.
        idx_a = int(self.rng.integers(0, len(_COLORS)))
        candidates = [i for i in range(len(_COLORS)) if i != idx_a]
        idx_b = candidates[int(self.rng.integers(0, len(candidates)))]
        color_a = _COLORS[idx_a]
        color_b = _COLORS[idx_b]
        # First cue is a Ball, second cue is a Key (deterministic so the
        # legend ordering is stable; randomness lives in colours and which
        # type the description asks about).
        cue_a = Ball(color=color_a)
        cue_b = Key(color=color_b)
        self._cue_pos_a = (2, max(1, corridor_y - 2))
        self._cue_pos_b = (2, min(self._grid_size - 2, corridor_y + 2))
        self._place_obj(*self._cue_pos_a, cue_a)
        self._place_obj(*self._cue_pos_b, cue_b)
        self._cue_visible = True

        # Pick which TYPE the fork asks about.
        if bool(self.rng.integers(0, 2)):
            self._target_type = Ball
            self._target_type_name = "ball"
            target_color = color_a
            distractor_color = color_b
        else:
            self._target_type = Key
            self._target_type_name = "key"
            target_color = color_b
            distractor_color = color_a

        correct_obj = self._target_type(color=target_color)
        wrong_obj = self._target_type(color=distractor_color)
        # Used by _task_description (says which TYPE to pick at the fork —
        # not which colour).
        self._memory_object_desc = ""
        self._cue_color = ""  # bind mode does not have a single cue colour
        self._cue_type = self._target_type
        self._place_fork_objects(corridor_y, target_x, correct_obj, wrong_obj)

    def _place_fork_objects(
        self,
        corridor_y: int,
        target_x: int,
        correct_obj: Any,
        wrong_obj: Any,
    ) -> None:
        if bool(self.rng.integers(0, 2)):
            self._correct_target_pos = (target_x, corridor_y - 1)
            self._wrong_target_pos = (target_x, corridor_y + 1)
        else:
            self._correct_target_pos = (target_x, corridor_y + 1)
            self._wrong_target_pos = (target_x, corridor_y - 1)
        self._correct_target_obj = correct_obj
        self._wrong_target_obj = wrong_obj
        self._fork_visible = False

    def _reveal_fork_objects(self) -> None:
        if self._fork_visible:
            return
        if self._correct_target_obj is None or self._wrong_target_obj is None:
            return
        self._place_obj(*self._correct_target_pos, self._correct_target_obj)
        self._place_obj(*self._wrong_target_pos, self._wrong_target_obj)
        self._fork_visible = True

    # -- top-level _generate_grid ---------------------------------------

    def _generate_grid(self, seed: int) -> None:
        corridor_y, _branch_x, target_x = self._carve_corridor()
        if self._mode == "type":
            self._gen_type_recall(corridor_y, target_x)
        elif self._mode == "color":
            self._gen_color_recall(corridor_y, target_x)
        elif self._mode == "bind":
            self._gen_bind_and_recall(corridor_y, target_x)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown _mode: {self._mode!r}")
        self._place_agent(1, corridor_y, DIR_RIGHT)

    # -- step ------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        choice: str | None = None
        if self.action_spec.names[action] == "PICKUP":
            front = self._front_pos()
            if front == self._correct_target_pos:
                choice = "correct"
            elif front == self._wrong_target_pos:
                choice = "wrong"

        if self._cue_visible:
            if self._mode == "bind":
                ax, ay = self._cue_pos_a
                bx, by = self._cue_pos_b
                self._place_obj(ax, ay, Wall())
                self._place_obj(bx, by, Wall())
            else:
                cx, cy = self._cue_pos
                self._place_obj(cx, cy, Wall())
            self._cue_visible = False
            self._reveal_fork_objects()

        obs, reward, terminated, truncated, info = super()._step(action)
        if choice == "correct":
            reward = self._reward_on_goal()
            terminated = True
            info["memory_choice"] = "correct"
            info["target_fetched"] = True
        elif choice == "wrong":
            # Additive: preserve any reward emitted by super()._step (e.g.,
            # step-time bonuses) before adding the wrong-choice penalty.
            reward += -1.0
            terminated = True
            info["memory_choice"] = "wrong"
        return obs, reward, terminated, truncated, info

    # -- prompt (NEVER leak the answer) ---------------------------------

    def _task_description(self) -> str:
        if self._mode == "type":
            return (
                "A long corridor briefly shows a single cue object near the "
                "start (a ball or a key, in some colour). The cue disappears "
                "after your first action. At the far fork, two objects of the "
                "same colour appear — one ball, one key. Pick up the one whose "
                "TYPE matches the cue you saw. Wrong choice ends the episode "
                "with -1 reward; correct choice rewards speed."
            )
        if self._mode == "color":
            return (
                "A long corridor briefly shows a single cue object (a ball or "
                "a key, in some colour) near the start. The cue disappears "
                "after your first action. At the far fork, two objects of the "
                "same TYPE appear in different colours. Pick up the one whose "
                "COLOUR matches the cue you saw. Wrong choice ends the episode "
                "with -1 reward; correct choice rewards speed."
            )
        if self._mode == "bind":
            return (
                f"A long corridor briefly shows TWO cue objects near the "
                f"start: a ball of one colour and a key of another. Both "
                f"disappear after your first action. At the far fork, two "
                f"{self._target_type_name}s of two different colours appear. "
                f"Pick up the {self._target_type_name} whose colour matches "
                f"the {self._target_type_name} you saw at the start. Wrong "
                f"choice ends the episode with -1 reward; correct choice "
                f"rewards speed."
            )
        return ""  # pragma: no cover


# ---------------------------------------------------------------------------
# Concrete envs.
# ---------------------------------------------------------------------------


class MiniGridMemoryS7Env(_MemoryBase):
    """Memory corridor in a 7x7 grid — pure type-recall."""

    _grid_size = 7
    _mode = "type"

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s7-v0"


class MiniGridMemoryS9Env(_MemoryBase):
    """Memory corridor in a 9x9 grid — pure colour-recall."""

    _grid_size = 9
    _mode = "color"

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s9-v0"


class MiniGridMemoryS11Env(_MemoryBase):
    """Memory corridor in an 11x11 grid — pure colour-recall."""

    _grid_size = 11
    _mode = "color"

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s11-v0"


class MiniGridMemoryS13Env(_MemoryBase):
    """Memory corridor in a 13x13 grid — bind-and-recall (two cues)."""

    _grid_size = 13
    _mode = "bind"

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s13-v0"


class MiniGridMemoryS17Env(_MemoryBase):
    """Memory corridor in a 17x17 grid — bind-and-recall (two cues)."""

    _grid_size = 17
    _mode = "bind"

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s17-v0"
