"""MiniHack Pray skill tasks.

Phase 3:
  * pray: randomise wall-gap and player/stairs positions per seed.
  * pray-distract: F6 liar-truth oracles — two altars, one heals, one hurts;
    two signs, one truthful one a liar. The sign glyphs are NEUTRAL (▣ / ▢)
    so that glyph identity is uncorrelated with truthfulness. A third tell —
    a wall inscription (✎) — names which glyph is the LIAR. The agent must
    read the inscription, then read both signs, then deduce the safe altar.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase

# Altar glyphs
ALTAR_GOOD = "♁"  # heals
ALTAR_BAD = "♆"  # damages
ALTAR_NAMES: dict[str, str] = {
    "♁": "altar (one is benevolent, one malevolent)",
    "♆": "altar (one is benevolent, one malevolent)",
}

# Neutral sign glyphs — no built-in truth/falsity semantics. The mapping
# (glyph → truthful) is randomised per-seed so that a fixed-glyph policy
# wins at chance level only.
SIGN_GLYPH_A = "▣"
SIGN_GLYPH_B = "▢"
SIGN_GLYPHS: tuple[str, str] = (SIGN_GLYPH_A, SIGN_GLYPH_B)
SIGN_NAMES: dict[str, str] = {
    SIGN_GLYPH_A: "sign (its claim may be true or false)",
    SIGN_GLYPH_B: "sign (its claim may be true or false)",
}

# Wall inscription glyph — the third tell. Stepping onto this tile yields a
# message that names which sign-glyph (▣ or ▢) is the LIAR.
INSCRIPTION_GLYPH = "✎"


class _PrayBase(MiniHackBase):
    _distract: bool = False
    _prayed: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._good_altar_pos: tuple[int, int] | None = None
        self._bad_altar_pos: tuple[int, int] | None = None
        self._truthful_sign_pos: tuple[int, int] | None = None
        self._lying_sign_pos: tuple[int, int] | None = None
        self._truthful_sign_target: tuple[int, int] | None = None
        self._lying_sign_target: tuple[int, int] | None = None
        self._sign_text_truthful: str = ""
        self._sign_text_lying: str = ""
        # Per-seed sign-glyph mapping (which glyph is on the truthful sign).
        self._truthful_sign_glyph: str = ""
        self._lying_sign_glyph: str = ""
        # Wall inscription: the third tell that disambiguates truthful sign.
        self._inscription_pos: tuple[int, int] = (0, 0)
        self._inscription_text: str = ""

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._prayed = False
        self._player_hp = 3

        if not self._distract:
            # Standard pray: vertical partition with random gap row.
            for y in range(1, 6):
                self._place_wall(3, y)
            gap_y = int(self.rng.integers(1, 6))
            self._grid[gap_y][3] = "·"
            # Player on left, random y
            py = int(self.rng.integers(1, 6))
            self._place_player(1, py)
            sy = int(self.rng.integers(1, 6))
            self._place_stairs(5, sy)
            return

        # Distract: F6 liar-truth oracles
        # Carve a 7x7 room with two altars (top and bottom) and two sign tiles.
        py = int(self.rng.integers(2, 5))
        self._place_player(1, py)
        sy = int(self.rng.integers(2, 5))
        self._place_stairs(5, sy)

        # Two altars: at columns 2 and 4, on row 1 and row 5 respectively
        altar_pos_1 = (3, 1)
        altar_pos_2 = (3, 5)
        # Random which is good
        good_first = bool(int(self.rng.integers(0, 2)))
        if good_first:
            self._good_altar_pos = altar_pos_1
            self._bad_altar_pos = altar_pos_2
        else:
            self._good_altar_pos = altar_pos_2
            self._bad_altar_pos = altar_pos_1
        gx, gy = self._good_altar_pos
        bx, by = self._bad_altar_pos
        self._grid[gy][gx] = ALTAR_GOOD
        self._grid[by][bx] = ALTAR_BAD

        # Two signs: at (2, 3) and (4, 3)
        sign_pos_1 = (2, 3)
        sign_pos_2 = (4, 3)
        truthful_first = bool(int(self.rng.integers(0, 2)))
        if truthful_first:
            self._truthful_sign_pos = sign_pos_1
            self._lying_sign_pos = sign_pos_2
        else:
            self._truthful_sign_pos = sign_pos_2
            self._lying_sign_pos = sign_pos_1
        # Each sign claims "altar at column X, row Y HEALS":
        # - truthful sign points to the GOOD altar (heals) — true claim
        # - lying sign    points to the BAD  altar (damages) — false claim
        self._truthful_sign_target = (gx, gy)
        self._lying_sign_target = (bx, by)

        # Decouple GLYPH from TRUTHFULNESS: random per-seed mapping.
        # If glyph_a_truthful, glyph A sits on the truthful sign, B on the
        # liar; otherwise swap. This makes a fixed-glyph policy chance-level.
        glyph_a_truthful = bool(int(self.rng.integers(0, 2)))
        if glyph_a_truthful:
            self._truthful_sign_glyph = SIGN_GLYPH_A
            self._lying_sign_glyph = SIGN_GLYPH_B
        else:
            self._truthful_sign_glyph = SIGN_GLYPH_B
            self._lying_sign_glyph = SIGN_GLYPH_A

        tsx, tsy = self._truthful_sign_pos
        lsx, lsy = self._lying_sign_pos
        self._grid[tsy][tsx] = self._truthful_sign_glyph
        self._grid[lsy][lsx] = self._lying_sign_glyph

        # Wall inscription — third tell. Place on the top wall row at
        # column 1 (an interior corner) by carving an inscription-floor.
        # Inscription text names which glyph is the LIAR (could equally
        # name the truthful one — randomised).
        self._inscription_pos = (1, 1)
        self._grid[1][1] = INSCRIPTION_GLYPH
        # The inscription names the LIAR's glyph so the agent must combine:
        # (1) the inscription's claim, with (2) the sign whose glyph it
        # excludes. The agent then trusts the OTHER sign's altar claim.
        self._inscription_text = (
            f"the LIAR sign carries the {self._lying_sign_glyph} glyph"
        )

    def _on_pray(self) -> None:
        # Standard pray: heals at any tile.
        if not self._distract:
            self._prayed = True
            self._player_hp = self._player_max_hp
            self._message += " You feel much better!"
            return
        # Distract: PRAY only works on the good altar tile.
        if self._player_pos == self._good_altar_pos:
            self._prayed = True
            self._player_hp = self._player_max_hp
            self._message += " You feel much better!"
        elif self._player_pos == self._bad_altar_pos:
            self._player_hp = max(0, self._player_hp - 3)
            self._message += " The altar burns you! (-3 HP)"
        else:
            self._message += " You are not on an altar."

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Allow movement onto altar / sign / inscription cells (walkable).
        name = self.action_spec.names[action]
        sign_glyphs_local = (self._truthful_sign_glyph, self._lying_sign_glyph)
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            walkable_special = (
                ALTAR_GOOD,
                ALTAR_BAD,
                INSCRIPTION_GLYPH,
            ) + sign_glyphs_local
            if terrain in walkable_special:
                self._player_pos = (nx, ny)
                if terrain == INSCRIPTION_GLYPH and self._distract:
                    self._message = (
                        f"The wall inscription reads: \"{self._inscription_text}.\""
                    )
                    return self._render_current_observation(), 0.0, False, False, {}
                if terrain in sign_glyphs_local and self._distract:
                    target = (
                        self._truthful_sign_target
                        if (nx, ny) == self._truthful_sign_pos
                        else self._lying_sign_target
                    )
                    if target is not None:
                        self._message = (
                            f"The sign reads: \"the altar at column {target[0]}, "
                            f"row {target[1]} HEALS the wounded.\""
                        )
                    return self._render_current_observation(), 0.0, False, False, {}
                self._message = ""
                return self._render_current_observation(), 0.0, False, False, {}

        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated and info.get("goal_reached") and not self._prayed:
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "You are too weak to descend safely. Pray first."
            obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if not self._distract:
            return obs
        # Make sure the legend describes the neutral sign glyphs and the
        # inscription tile (so the agent knows they exist + what they do).
        legend = obs.legend
        for glyph in SIGN_GLYPHS:
            if glyph in obs.grid and glyph not in legend:
                legend = legend + f"\n{glyph} — " + SIGN_NAMES[glyph]
        if INSCRIPTION_GLYPH in obs.grid and INSCRIPTION_GLYPH not in legend:
            legend = (
                legend
                + f"\n{INSCRIPTION_GLYPH} — wall inscription (step on to read)"
            )
        for glyph in (ALTAR_GOOD, ALTAR_BAD):
            if glyph in obs.grid and glyph not in legend:
                legend = legend + f"\n{glyph} — " + ALTAR_NAMES[glyph]
        return GridObservation(
            grid=obs.grid,
            legend=legend,
            hud=obs.hud,
            message=obs.message,
        )

    def _task_description(self) -> str:
        if self._distract:
            return (
                "You are near death. Two altars (♁/♆) stand in the room — one "
                "heals when you PRAY on it, the other damages. Two signs in the "
                "middle each carry one of two NEUTRAL glyphs (▣ / ▢) and each "
                "claims 'the altar at column X, row Y heals'; ONE sign tells "
                "the truth, the other LIES — but WHICH glyph is the liar is "
                "decided per-seed and the glyph itself carries no clue. A wall "
                "inscription (✎) names which glyph belongs to the liar; step "
                "on it to read its message. Cross-reference the inscription "
                "with the two signs to identify the safe altar, walk to it "
                "and PRAY. Then reach the stairs (⇣). Reward: +1 stairs, -1 "
                "death."
            )
        return (
            "You are near death (low HP). Use PRAY to heal yourself, "
            "then navigate to the stairs (⇣). Reward: +1 stairs, -1 death."
        )


class MiniHackPrayEnv(_PrayBase):
    """MiniHack Pray: pray to heal low HP, then reach stairs."""

    def env_id(self) -> str:
        return "glyphbench/minihack-pray-v0"


class MiniHackPrayDistractEnv(_PrayBase):
    """MiniHack Pray (Distract): F6 liar-truth altars + signs."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-pray-distract-v0"
