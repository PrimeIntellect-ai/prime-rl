"""MiniHack HideNSeek environments.

Mechanic: G3 stalker. A kobold patrols a pillar maze. The kobold ONLY
steps when it has line-of-sight (LOS) to the agent — otherwise it idles.
The agent must use the wall pillars to break LOS, slip past, and reach
the stairs.

Variants (size + extras):
  * hidenseek-v0       9x9 base
  * hidenseek-mapped-v0 12x12 (bigger maze, distinct layout)
  * hidenseek-big-v0   15x15 (largest)
  * hidenseek-lava-v0  12x12 + lava patches to dodge
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD


class _HideNSeekBase(MiniHackBase):
    _size: int = 9
    _has_lava: bool = False
    _pillar_step: int = 3  # spacing between vertical pillar columns

    def _generate_level(self, seed: int) -> None:
        s = self._size
        self._init_grid(s, s)
        cy = s // 2

        # Vertical wall pillars with a gap at the middle row.
        for i in range(2, s - 2, self._pillar_step):
            for y in range(1, s - 1):
                if y != cy:
                    self._place_wall(i, y)

        # Lava patches (only in lava variant)
        if self._has_lava:
            for y in (1, s - 2):
                for x in range(1, s - 1):
                    if self._grid[y][x] == "·" and self.rng.random() < 0.25:
                        self._place_lava(x, y)

        # I5: randomise player start within the leftmost interior column —
        # any free floor cell in column x=1 is a valid start.
        leftmost_candidates = [
            (1, y)
            for y in range(1, s - 1)
            if self._grid[y][1] == "·"
        ]
        # Fall back to (1, 1) if (theoretically) no candidates.
        if leftmost_candidates:
            idx = int(self.rng.integers(0, len(leftmost_candidates)))
            px, py = leftmost_candidates[idx]
        else:
            px, py = 1, 1
        self._place_player(px, py)

        # Stairs at SE corner (clear out lava there if any)
        sx, sy = s - 2, s - 2
        if self._grid[sy][sx] != "·":
            self._grid[sy][sx] = "·"
        self._place_stairs(sx, sy)

        # I5: randomise stalker kobold within the midline gap row (y=cy).
        # The kobold MUST stay on the midline so the stalker mechanic is
        # unchanged (gap row gives it a clear east-west corridor).
        midline_candidates = [
            (x, cy)
            for x in range(2, s - 1)
            if self._grid[cy][x] == "·"
            and (x, cy) != (px, py)
            and (x, cy) != (sx, sy)
        ]
        if midline_candidates:
            idx = int(self.rng.integers(0, len(midline_candidates)))
            kx, ky = midline_candidates[idx]
        else:
            kx, ky = (s - 1) // 2, cy
        self._spawn_creature(KOBOLD, kx, ky)

    def _move_monsters(self) -> None:
        """Stalker AI: move only when LOS to player exists; else idle."""
        px, py = self._player_pos
        for c in self._creatures:
            if c.hp <= 0 or c.ctype.ai != "hostile":
                continue
            if not self._has_line_of_sight((c.x, c.y), (px, py)):
                # No LOS → kobold idles (does NOT move)
                continue
            # LOS → chase
            dx = 0 if c.x == px else (1 if c.x < px else -1)
            dy = 0 if c.y == py else (1 if c.y < py else -1)
            nx, ny = c.x + dx, c.y + dy
            if (nx, ny) == (px, py):
                dmg = max(1, c.ctype.damage)
                self._player_hp -= dmg
                self._message += f" The {c.ctype.name} hits you! (-{dmg} HP)"
                continue
            if (
                self._is_walkable_for_monster(nx, ny)
                and self._creature_at(nx, ny) is None
            ):
                c.x, c.y = nx, ny

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            if (
                self._goal_pos
                and (nx, ny) == self._goal_pos
                and self._is_walkable(nx, ny)
            ):
                self._player_pos = (nx, ny)
                self._message = "You reach the stairs. You descend."
                return self._render_current_observation(), 1.0, True, False, {
                    "goal_reached": True,
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
            monster = self._creature_at(nx, ny)
            if monster is not None:
                dmg = max(1, monster.ctype.damage)
                self._player_hp -= dmg
                self._message = (
                    f"The {monster.ctype.name} blocks your path; you cannot "
                    f"fight the {monster.ctype.name}. (-{dmg} HP)"
                )
                if self._player_hp <= 0:
                    self._message += " You die."
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "combat"
                    }
                self._move_monsters()
                if self._player_hp <= 0:
                    self._message = (self._message + " You die.").strip()
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "monster"
                    }
                return self._render_current_observation(), 0.0, False, False, {
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
        return super()._step(action)

    def _movement_prompt(self) -> str:
        return (
            "8 directional movement: N, S, E, W, NE, NW, SE, SW. Moving into "
            "a wall does nothing. In this task, moving into the kobold does "
            "not attack it: the kobold blocks you and damages you, then the "
            "monster turn still resolves."
        )

    def _task_description(self) -> str:
        parts = [
            f"Navigate a {self._size}x{self._size} pillar maze to reach the "
            "stairs (⇣). A kobold (k) stalks the maze BUT only moves when it "
            "has direct line-of-sight to you (no walls between). Use the wall "
            "pillars to break line-of-sight: the kobold freezes whenever you "
            "step behind a pillar. The kobold cannot be killed or fought as "
            "a shortcut; avoid contact and slip past it."
        ]
        if self._has_lava:
            parts.append(
                "Lava (♨) patches are scattered around — stepping in lava is "
                "fatal."
            )
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)


class MiniHackHideNSeekEnv(_HideNSeekBase):
    """Base 9x9 LOS-stalker maze."""

    _size = 9
    _pillar_step = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-hidenseek-v0"


class MiniHackHideNSeekMappedEnv(_HideNSeekBase):
    """12x12 LOS-stalker (mid-size, denser pillars)."""

    _size = 12
    _pillar_step = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-hidenseek-mapped-v0"


class MiniHackHideNSeekLavaEnv(_HideNSeekBase):
    """12x12 LOS-stalker with lava hazards."""

    _size = 12
    _pillar_step = 3
    _has_lava = True

    def env_id(self) -> str:
        return "glyphbench/minihack-hidenseek-lava-v0"


class MiniHackHideNSeekBigEnv(_HideNSeekBase):
    """15x15 LOS-stalker (largest maze)."""

    _size = 15
    _pillar_step = 4

    def env_id(self) -> str:
        return "glyphbench/minihack-hidenseek-big-v0"
