"""MiniHack Read skill tasks.

Phase 3:
  * read: randomise scroll + wall-gap + stair positions.
  * read-distract: L2 Simon-says — three scrolls on the floor, each with
    its own destination. The HUD message log shows a flashing pattern that
    encodes which scroll is the safe teleport. Wrong scroll → teleport to
    lava.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.items import SCROLL_TELEPORT, Item

# Three numbered teleport scrolls used by the distract variant.
# Distinct single-codepoint glyphs keep the legend unambiguous.
SCROLL_TELEPORT_A = Item("scroll of teleportation A", "A", "scroll")
SCROLL_TELEPORT_B = Item("scroll of teleportation B", "B", "scroll")
SCROLL_TELEPORT_C = Item("scroll of teleportation C", "C", "scroll")


class _ReadBase(MiniHackBase):
    _distract: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._safe_scroll_name: str = ""
        self._pattern_step: int = 0
        self._pattern_glyphs: list[str] = []
        self._lava_pos: tuple[int, int] | None = None

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        # Build wall column at random x in [2, 4] (so right half is x in [3, 5])
        wall_x = int(self.rng.integers(2, 5))
        for y in range(1, 6):
            self._place_wall(wall_x, y)

        # Player at random position in the left half (excluding wall)
        left_candidates = [
            (x, y) for y in range(1, 6) for x in range(1, wall_x)
        ]
        idx = int(self.rng.integers(0, len(left_candidates)))
        px, py = left_candidates[idx]
        self._place_player(px, py)

        # Stairs at random position in the right half
        right_candidates = [
            (x, y) for y in range(1, 6) for x in range(wall_x + 1, 6)
        ]
        idx = int(self.rng.integers(0, len(right_candidates)))
        self._place_stairs(*right_candidates[idx])

        # Place scroll(s) in left half
        scroll_candidates = [
            c for c in left_candidates if c != (px, py)
        ]
        if not self._distract:
            scroll_pos = scroll_candidates[int(self.rng.integers(0, len(scroll_candidates)))]
            self._place_item(*scroll_pos, SCROLL_TELEPORT)
        else:
            # Place 3 scrolls; one is safe, others teleport to lava.
            scrolls = [SCROLL_TELEPORT_A, SCROLL_TELEPORT_B, SCROLL_TELEPORT_C]
            shuffled = [
                scroll_candidates[i]
                for i in self.rng.permutation(len(scroll_candidates))[:3]
            ]
            for scroll, pos in zip(scrolls, shuffled, strict=True):
                self._place_item(*pos, scroll)
            # Random pick which scroll is safe
            safe_idx = int(self.rng.integers(0, 3))
            self._safe_scroll_name = scrolls[safe_idx].name
            # Pattern: encode safe scroll via a per-step glyph cycle in HUD.
            # Pattern: 'A' for A, 'B' for B, 'C' for C — repeats over time.
            target_letter = self._safe_scroll_name[-1]  # last char e.g. "A"
            self._pattern_glyphs = [target_letter, "·", "·"]
            self._pattern_step = 0
            # Place lava on the right side (not on stairs)
            lava_candidates = [c for c in right_candidates if c != self._goal_pos]
            self._lava_pos = lava_candidates[
                int(self.rng.integers(0, len(lava_candidates)))
            ]
            lx, ly = self._lava_pos
            self._place_lava(lx, ly)

    def _on_read_scroll(self, scroll: Item) -> None:
        if not self._distract:
            if "teleportation" in scroll.name and self._goal_pos is not None:
                self._player_pos = self._goal_pos
                self._message += " You feel a wrenching sensation!"
            return
        # Distract: teleport to stairs only if it's the safe scroll;
        # else teleport to lava (death).
        if scroll.name == self._safe_scroll_name and self._goal_pos:
            self._player_pos = self._goal_pos
            self._message += " You feel a wrenching sensation — safe!"
        elif self._lava_pos:
            self._player_pos = self._lava_pos
            self._message += " You teleport ONTO lava and burn to death!"
            self._player_hp = 0

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        if self._distract:
            self._pattern_step += 1
        return super()._step(action)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._distract and self._pattern_glyphs:
            current = self._pattern_glyphs[
                self._pattern_step % len(self._pattern_glyphs)
            ]
            extra = f"    Pattern (Simon-says): {current}"
            new_hud = obs.hud + extra
            return GridObservation(
                grid=obs.grid,
                legend=obs.legend,
                hud=new_hud,
                message=obs.message,
            )
        return obs

    def _task_description(self) -> str:
        if self._distract:
            return (
                "A wall blocks your path to the stairs (⇣). Three scrolls of "
                "teleportation (A/B/C) are scattered. Only one safely teleports "
                "to the stairs; the other two send you to LAVA (instant death). "
                "The HUD shows a 'Pattern' that cycles A/·/· (or B/·/· etc); "
                "the letter in the cycle is the safe scroll's name. PICKUP the "
                "right scroll and READ it. Reward: +1 stairs, -1 death."
            )
        return (
            "A wall blocks your path to the stairs (⇣). "
            "Pick up the scroll of teleportation (?) and READ it to teleport "
            "past the wall. Reward: +1 stairs, -1 death."
        )


class MiniHackReadEnv(_ReadBase):
    """MiniHack Read: read a scroll of teleportation to bypass a wall."""

    def env_id(self) -> str:
        return "glyphbench/minihack-read-v0"


class MiniHackReadDistractEnv(_ReadBase):
    """MiniHack Read (Distract): L2 Simon-says scroll picker."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-read-distract-v0"
