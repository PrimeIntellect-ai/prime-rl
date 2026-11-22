"""Nim: classic game theory -- take objects from piles.

Player who takes the last object loses (misere Nim).

Gym IDs:
  glyphbench/classics-nim-easy-v0   (piles 3, 5, 7)
  glyphbench/classics-nim-hard-v0   (piles 5, 7, 9)
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Action spec builder
# ---------------------------------------------------------------------------


def _make_nim_action_spec(pile_sizes: tuple[int, ...]) -> ActionSpec:
    """Create TAKE_pile_count actions for all valid (pile, count) pairs.

    E.g. for pile 0 of size 3: TAKE_0_1, TAKE_0_2, TAKE_0_3.
    """
    names: list[str] = []
    descriptions: list[str] = []
    for p, sz in enumerate(pile_sizes):
        for k in range(1, sz + 1):
            names.append(f"TAKE_{p}_{k}")
            descriptions.append(f"take {k} object(s) from pile {p}")
    return ActionSpec(names=tuple(names), descriptions=tuple(descriptions))


# ---------------------------------------------------------------------------
# Base Nim Env
# ---------------------------------------------------------------------------


class _NimBase(BaseGlyphEnv):
    """Nim: take objects from piles. Whoever takes the last object loses."""

    noop_action_name: str = "TAKE_0_1"

    _initial_piles: tuple[int, ...] = (3, 5, 7)
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        self.action_spec = _make_nim_action_spec(self._initial_piles)
        super().__init__(max_turns=max_turns)
        self._piles: list[int] = []
        self._message: str = ""

    def env_id(self) -> str:
        return f"glyphbench/classics-nim-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # AI opponent (optimal Nim / XOR strategy)
    # ------------------------------------------------------------------

    def _ai_move(self) -> tuple[int, int] | None:
        """Optimal misere Nim AI using XOR (nim-sum) strategy."""
        piles = self._piles
        total = sum(piles)
        if total == 0:
            return None

        nim_sum = 0
        for p in piles:
            nim_sum ^= p

        # Count piles with more than 1 object
        big_piles = sum(1 for p in piles if p > 1)

        if big_piles == 0:
            # Only piles of size 0 or 1 left. In misere Nim, the mover wants
            # to leave an odd number of 1-piles to the opponent.
            ones = sum(1 for p in piles if p == 1)
            if ones % 2 == 1:
                for i, p in enumerate(piles):
                    if p == 1:
                        return (i, 1)
            else:
                for i, p in enumerate(piles):
                    if p > 0:
                        return (i, 1)
            return None

        if nim_sum == 0:
            # Losing position for the mover -- make a random valid move
            valid = [(i, k) for i, p in enumerate(piles) for k in range(1, p + 1)]
            if valid:
                idx = int(self.rng.integers(0, len(valid)))
                return valid[idx]
            return None

        # Winning strategy: find a pile to reduce so nim-sum becomes 0
        for i, p in enumerate(piles):
            target = p ^ nim_sum
            if target < p:
                take = p - target
                # Misere adjustment: if this would leave all piles <= 1
                remaining = list(piles)
                remaining[i] = target
                if all(r <= 1 for r in remaining):
                    # We want to leave an ODD number of 1-piles
                    ones_after = sum(1 for r in remaining if r == 1)
                    if ones_after % 2 == 0:
                        # Adjust: take one more or one less if possible
                        if target > 0 and take + 1 <= p:
                            take += 1
                        elif take > 1:
                            take -= 1
                return (i, take)

        # Fallback: random
        valid = [(i, k) for i, p in enumerate(piles) for k in range(1, p + 1)]
        if valid:
            idx = int(self.rng.integers(0, len(valid)))
            return valid[idx]
        return None

    # ------------------------------------------------------------------
    # Parse action
    # ------------------------------------------------------------------

    def _parse_action(self, action: int) -> tuple[int, int]:
        """Convert action index to (pile, count)."""
        name = self.action_spec.names[action]
        parts = name.split("_")
        pile = int(parts[1])
        count = int(parts[2])
        return pile, count

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._piles = list(self._initial_piles)
        self._message = "Your turn. Take objects from a pile."
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        pile, count = self._parse_action(action)

        # Validate move
        if pile >= len(self._piles) or count > self._piles[pile] or count < 1:
            self._message = (
                f"Invalid move: pile {pile} has {self._piles[pile]} objects. You lose."
            )
            info["outcome"] = "invalid_move"
            return self._render_current_observation(), -1.0, True, False, info

        # Player takes
        self._piles[pile] -= count
        self._message = f"You took {count} from pile {pile}."

        # Check if player took the last object (player loses in misere)
        if sum(self._piles) == 0:
            self._message += " You took the last object. You lose!"
            info["outcome"] = "loss"
            return self._render_current_observation(), -1.0, True, False, info

        # AI move
        ai_move = self._ai_move()
        if ai_move is not None:
            ai_pile, ai_count = ai_move
            self._piles[ai_pile] -= ai_count
            self._message += f" Opponent took {ai_count} from pile {ai_pile}."

            if sum(self._piles) == 0:
                self._message += " Opponent took the last object. You win!"
                info["outcome"] = "win"
                return self._render_current_observation(), 1.0, True, False, info

        info["piles"] = list(self._piles)
        return self._render_current_observation(), 0.0, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        # Render piles as rows of objects
        max_pile = max(self._initial_piles)
        num_piles = len(self._piles)
        # Grid: one row per pile, width = max_pile
        grid = make_empty_grid(max_pile, num_piles, SYM_EMPTY)

        SYM_OBJ = "\u25a0"   # ■
        SYM_EMPTY_CELL = "\u00b7"  # ·

        for r in range(num_piles):
            for c in range(max_pile):
                if c < self._piles[r]:
                    grid[r][c] = SYM_OBJ
                else:
                    grid[r][c] = SYM_EMPTY_CELL

        legend = build_legend({
            SYM_OBJ: "object in pile",
            SYM_EMPTY_CELL: "empty",
        })

        # HUD shows pile counts
        pile_info = "  ".join(f"Pile {i}: {self._piles[i]}" for i in range(num_piles))
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"{pile_info}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def system_prompt(self) -> str:
        piles_str = ", ".join(str(p) for p in self._initial_piles)
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Play Nim against an optimal AI. Take objects from piles. The player "
            "who takes the LAST object LOSES (misere Nim).\n\n"
            "RULES\n"
            f"- There are {len(self._initial_piles)} piles with sizes: {piles_str}.\n"
            "- On your turn, you must take at least 1 object from exactly one pile.\n"
            "- You can take any number of objects from that pile (up to its current size).\n"
            "- The player forced to take the last object loses.\n"
            "- Actions are TAKE_pile_count (e.g. TAKE_0_2 takes 2 from pile 0).\n"
            "- Invalid moves lose immediately.\n"
            "- Reward: +1 for a win, -1 for a loss.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

SYM_EMPTY = "\u00b7"


class NimEasyEnv(_NimBase):
    _initial_piles = (3, 5, 7)
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class NimHardEnv(_NimBase):
    _initial_piles = (5, 7, 9)
    _difficulty = "hard"

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
