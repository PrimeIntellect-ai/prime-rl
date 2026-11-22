"""MiniHack KeyRoom environments.

A room with a locked door (⊞) blocking the path to the stairs.
The agent must collect a brass key first, then move into the door to unlock
it (⊞ → ·), then proceed to the stairs.

Phase 3: each variant gets a distinct twist:
  * s5: randomised positions + a decoy key.
  * s15: randomised + B3 cul-de-sac (one fake door leading to a dead-end room).
  * dark-s5: randomised + A6 counted-memory of torches passed.
  * dark-s15: randomised + E1 torch budget (limited torches; without torch
              you see only the adjacent 3x3).
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.items import BRASS_KEY, Item

TORCH_GLYPH = "ψ"  # torch on floor (also used as a counted glyph)
ANSWER_GLYPHS: tuple[str, str, str, str, str, str] = (
    "➀", "➁", "➂", "➃", "➄", "➅",
)
ANSWER_NAMES: dict[str, str] = {
    "➀": "answer 1",
    "➁": "answer 2",
    "➂": "answer 3",
    "➃": "answer 4",
    "➄": "answer 5",
    "➅": "answer 6",
}

TORCH_ITEM = Item("torch", "↑", "torch")  # carried torch in inventory
DECOY_KEY = Item("tin key (decoy)", "⌐", "key")


class _KeyRoomBase(MiniHackBase):
    """Base class for KeyRoom environments."""

    _room_interior: int = 5
    _is_dark: bool = False
    _has_culdesac: bool = False
    _has_decoy_key: bool = False
    _has_counted_memory: bool = False
    _has_torch_budget: bool = False
    _locked_door_pos: tuple[int, int] = (0, 0)

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._fake_door_pos: tuple[int, int] | None = None
        self._dead_end_room_x_start: int = 0
        self._dead_end_room_x_end: int = 0
        # Counted-memory state
        self._torches_to_count: int = 0
        self._torches_passed: int = 0
        self._answer_positions: dict[tuple[int, int], int] = {}
        # Torch-budget state
        self._torch_charges: int = 0  # remaining torch charges
        self._torch_lit_turns: int = 0

    def _generate_level(self, seed: int) -> None:
        ri = self._room_interior
        w = ri + 2
        h = ri + 2

        # cul-de-sac variant adds an extra room to the south
        if self._has_culdesac:
            h = ri + 2 + (ri // 2 + 2)
        self._init_grid(w, h)
        self._dark = self._is_dark
        self._torch_lit_turns = 0

        # Vertical partition wall (random x in middle 60% of interior)
        partition_x = w // 2
        for y in range(1, ri + 1):
            self._place_wall(partition_x, y)

        # Player y on left side of partition
        py = int(self.rng.integers(1, ri + 1))
        self._place_player(1, py)

        # Door y at random row (must be on the partition wall)
        door_y = int(self.rng.integers(1, ri + 1))
        self._place_door(partition_x, door_y)
        self._locked_door_pos = (partition_x, door_y)

        # Key at random position in the left half (excluding player + door)
        left_candidates = [
            (x, y)
            for y in range(1, ri + 1)
            for x in range(1, partition_x)
            if (x, y) != (1, py)
        ]
        idx = int(self.rng.integers(0, len(left_candidates)))
        kx, ky = left_candidates[idx]
        self._place_item(kx, ky, BRASS_KEY)

        if self._has_decoy_key:
            decoy_candidates = [
                (x, y)
                for x, y in left_candidates
                if (x, y) != (kx, ky)
            ]
            if decoy_candidates:
                didx = int(self.rng.integers(0, len(decoy_candidates)))
                dx, dy = decoy_candidates[didx]
                self._place_item(dx, dy, DECOY_KEY)

        # Stairs at random position on the right side
        right_candidates = [
            (x, y)
            for y in range(1, ri + 1)
            for x in range(partition_x + 1, w - 1)
        ]
        idx = int(self.rng.integers(0, len(right_candidates)))
        sx, sy = right_candidates[idx]
        self._place_stairs(sx, sy)

        # Variant-specific decoration
        if self._has_culdesac:
            # Add a fake door to a dead-end room to the south.
            # Seal the row below the main room so the side room cannot be
            # reached by walking around the partition.
            tunnel_x = int(self.rng.integers(1, partition_x))
            dead_end_y0 = ri + 2
            dead_end_y1 = h - 1

            for x in range(1, w - 1):
                self._grid[ri + 1][x] = "█"

            # Carve a one-cell tunnel below the fake door, then the room.
            self._grid[ri + 1][tunnel_x] = "·"
            self._grid[dead_end_y0][tunnel_x] = "·"
            # Carve dead-end room (3 wide centred on tunnel_x)
            for y in range(dead_end_y0 + 1, dead_end_y1):
                for x in range(max(1, tunnel_x - 1), min(w - 1, tunnel_x + 2)):
                    self._grid[y][x] = "·"
            # Place fake door at junction
            self._place_door(tunnel_x, ri + 1)
            self._fake_door_pos = (tunnel_x, ri + 1)
            self._dead_end_room_x_start = max(1, tunnel_x - 1)
            self._dead_end_room_x_end = min(w - 1, tunnel_x + 2)

        if self._has_counted_memory:
            # Place 1-4 torches in the LEFT half of the room
            torch_count = int(self.rng.integers(1, 5))
            self._torches_to_count = torch_count
            self._torches_passed = 0
            torch_candidates = [
                (x, y)
                for y in range(1, ri + 1)
                for x in range(1, partition_x)
                if (x, y) != (1, py) and (x, y) != (kx, ky)
            ]
            tindices = list(self.rng.permutation(len(torch_candidates)))[:torch_count]
            for ti in tindices:
                tx, ty = torch_candidates[ti]
                self._grid[ty][tx] = TORCH_GLYPH

            # Place 6 numbered answer plates on the RIGHT half (replacing some floor)
            right_candidates_for_plates = [
                (x, y)
                for y in range(1, ri + 1)
                for x in range(partition_x + 1, w - 1)
                if (x, y) != (sx, sy)
            ]
            shuffled = list(self.rng.permutation(len(right_candidates_for_plates)))
            self._answer_positions = {}
            for ai, slot_idx in enumerate(shuffled[:len(ANSWER_GLYPHS)]):
                pos = right_candidates_for_plates[slot_idx]
                gx, gy = pos
                self._grid[gy][gx] = ANSWER_GLYPHS[ai]
                self._answer_positions[pos] = ai + 1

        if self._has_torch_budget:
            # Player gets 2 torch charges; each STEP without a torch held
            # uses radius 1, but holding/applying a torch extends to radius 3
            # for the next 8 turns.
            self._torch_charges = 2
            # Place 1 torch on the floor that the player can pick up.
            torch_candidates = [
                (x, y)
                for y in range(1, ri + 1)
                for x in range(1, partition_x)
                if (x, y) != (1, py) and (x, y) != (kx, ky)
            ]
            if torch_candidates:
                tidx = int(self.rng.integers(0, len(torch_candidates)))
                tx, ty = torch_candidates[tidx]
                self._place_item(tx, ty, TORCH_ITEM)

    def _has_key(self) -> bool:
        return any(item.name == BRASS_KEY.name for item in self._inventory)

    def _has_torch(self) -> bool:
        return any(item.name == TORCH_ITEM.name for item in self._inventory)

    def _tick_torch_light(self) -> None:
        if self._has_torch_budget and self._torch_lit_turns > 0:
            self._torch_lit_turns -= 1

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]

        # Counted-memory: stepping on an answer plate ends episode.
        if self._has_counted_memory and name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            if terrain in ANSWER_GLYPHS:
                # Answer plates only count if the agent has the key
                # (otherwise this is "premature": gate is sealed).
                self._player_pos = (nx, ny)
                if not self._has_key():
                    self._message = (
                        f"You step on {ANSWER_NAMES[terrain]}. The seal is "
                        "still active — find the brass key first."
                    )
                    return self._render_current_observation(), 0.0, False, False, {}
                value = self._answer_positions.get((nx, ny), 0)
                if value == self._torches_to_count:
                    self._message = (
                        f"Correct! You counted {self._torches_to_count} torches."
                    )
                    return self._render_current_observation(), 1.0, True, False, {
                        "goal_reached": True,
                    }
                self._message = (
                    f"Wrong count. You said {value}, the answer was "
                    f"{self._torches_to_count}."
                )
                return self._render_current_observation(), -1.0, True, False, {
                    "wrong_answer": True
                }
            elif terrain == TORCH_GLYPH:
                self._player_pos = (nx, ny)
                self._grid[ny][nx] = "·"
                self._torches_passed += 1
                # The env tracks _torches_passed internally for grading, but
                # the message MUST NOT echo the running tally — that would
                # defeat the A6 counted-memory mechanic.
                self._message = (
                    "You walk past a torch (ψ); it burns out behind you."
                )
                return self._render_current_observation(), 0.0, False, False, {}
            elif terrain == "⇣":
                self._player_pos = (nx, ny)
                self._message = (
                    "The stairs are sealed. Step on the numbered answer plate."
                )
                return self._render_current_observation(), 0.0, False, False, {
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }

        # Torch-budget: handle APPLY action to use a torch charge
        if self._has_torch_budget and name == "APPLY" and self._has_torch() and self._torch_charges > 0:
            self._torch_charges -= 1
            # Grant 8 turns of extended vision via _levitating_turns proxy
            # — we use a separate counter.
            self._torch_lit_turns = 8
            self._message = (
                f"You light the torch. ({self._torch_charges} charges left)"
            )
            # Don't consume the action; render and return
            return self._render_current_observation(), 0.0, False, False, {}

        # Door handling
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            if terrain == "⊞":
                if not self._has_key():
                    self._message = "The door is locked. You need the brass key."
                    return self._finish_turn()
                # Check if it's a fake/decoy door
                if (
                    self._has_culdesac
                    and self._fake_door_pos is not None
                    and (nx, ny) == self._fake_door_pos
                ):
                    self._grid[ny][nx] = "·"
                    self._player_pos = (nx, ny)
                    self._message = (
                        "You unlock the door — but it leads to a dead-end side "
                        "room. The real exit is elsewhere."
                    )
                    return self._finish_turn(
                        info={
                            "fake_door": True,
                            "player_pos": self._player_pos,
                            "hp": self._player_hp,
                        }
                    )
                self._grid[ny][nx] = "·"
                self._player_pos = (nx, ny)
                self._message = "You unlock and open the door with the brass key."
                return self._finish_turn(
                    info={"player_pos": self._player_pos, "hp": self._player_hp}
                )

        # Decrement torch_lit_turns each step
        self._tick_torch_light()

        return super()._step(action)

    def _finish_turn(
        self,
        *,
        reward: float = 0.0,
        terminated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info = {} if info is None else info
        self._tick_torch_light()

        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = (
                "combat" if "hit" in self._message.lower() else "hazard"
            )
            return self._render_current_observation(), -1.0, terminated, False, info

        self._move_monsters()

        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = "monster"
            return self._render_current_observation(), -1.0, terminated, False, info

        if (
            self._has_counted_memory
            and self._goal_pos
            and self._player_pos == self._goal_pos
        ):
            self._message = "The stairs are sealed. Step on the numbered answer plate."
        elif self._goal_pos and self._player_pos == self._goal_pos:
            terminated = True
            reward = 1.0
            self._message = "You reach the stairs. You descend."
            info["goal_reached"] = True

        info["player_pos"] = self._player_pos
        info["hp"] = self._player_hp
        return self._render_current_observation(), reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        # Torch-budget: temporarily widen vision radius if torch is lit.
        original_radius = self._vision_radius
        if self._has_torch_budget and self._torch_lit_turns > 0:
            self._vision_radius = 3
        try:
            obs = super()._render_current_observation()
        finally:
            self._vision_radius = original_radius

        # Append variant-specific HUD info
        extra = ""
        if self._has_counted_memory:
            # A6 counted-memory: HUD MUST NOT expose the running tally —
            # the agent must count torches themselves.
            extra = "    Walked past torches: count them yourself"
        elif self._has_torch_budget:
            lit = self._torch_lit_turns
            if self._has_torch():
                extra = (
                    f"    Torch charges: {self._torch_charges}    "
                    f"Torch lit: {'yes (' + str(lit) + ' turns)' if lit > 0 else 'no'}"
                )
            else:
                extra = f"    Torch: not carried ({self._torch_charges} charges after pickup)"
        new_hud = obs.hud + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    def _task_description(self) -> str:
        if self._has_counted_memory:
            parts = [
                "A locked door blocks the answer room. Find the brass key (() "
                "in the left half, stand on it and PICKUP, then move into the "
                "door to unlock it."
            ]
        else:
            parts = [
                "A locked door blocks the stairs (⇣). Find the brass key (() in "
                "the left half, stand on it and PICKUP, then move into the door "
                "to unlock it and reach the stairs."
            ]
        if self._has_culdesac:
            parts.append(
                "Beware: a SECOND door leads to a dead-end side room — "
                "unlocking it gives no progress and wastes a turn."
            )
        if self._has_decoy_key:
            parts.append(
                "A tin key (⌐) is a decoy; only the brass key (() opens the door."
            )
        if self._has_counted_memory:
            parts.append(
                "Torches (ψ) on the left side burn out as you walk past them. "
                "COUNT them as you pass. After collecting the key, on the "
                "right side step on the numbered plate (➀..➅) matching the "
                "torch count. The numbered plate, not the stairs, is the terminal goal."
            )
        if self._has_torch_budget:
            parts.append(
                "The room is dark; you can only see your immediate "
                "neighbours. Pick up a portable torch on the floor (↑) and "
                "use APPLY to light it for 8 turns of extended vision; you "
                "have only 2 charges total."
            )
        elif self._is_dark:
            parts.append(
                "The room is dark — you can only see tiles adjacent to you."
            )
        if self._has_counted_memory:
            parts.append("Reward: +1 on the correct answer plate, -1 on wrong answer.")
        else:
            parts.append("Reward: +1 on success.")
        return " ".join(parts)


class MiniHackKeyRoomS5Env(_KeyRoomBase):
    """5x5 KeyRoom with randomised positions and a decoy key."""

    _room_interior = 5
    _has_decoy_key = True

    def env_id(self) -> str:
        return "glyphbench/minihack-keyroom-s5-v0"


class MiniHackKeyRoomS15Env(_KeyRoomBase):
    """15x15 KeyRoom with randomised positions + B3 cul-de-sac."""

    _room_interior = 15
    _has_culdesac = True

    def env_id(self) -> str:
        return "glyphbench/minihack-keyroom-s15-v0"


class MiniHackKeyRoomDarkS5Env(_KeyRoomBase):
    """5x5 dark KeyRoom + A6 counted-memory of torches."""

    _room_interior = 5
    _is_dark = True
    _has_counted_memory = True

    def env_id(self) -> str:
        return "glyphbench/minihack-keyroom-dark-s5-v0"


class MiniHackKeyRoomDarkS15Env(_KeyRoomBase):
    """15x15 dark KeyRoom + E1 torch budget."""

    _room_interior = 15
    _is_dark = True
    _has_torch_budget = True

    def env_id(self) -> str:
        return "glyphbench/minihack-keyroom-dark-s15-v0"
