"""Match-3 gem-swapping puzzle game.

Phase 3 audit fix (2026-05-03): mechanic M1 (permutation goal).
- Action space restricted to 60 swap-pair encodings (RIGHT/DOWN only).
- Each round shows a 2x2 target pattern that must appear on the board for
  bonus reward; on completion a fresh pattern is rolled.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIZE = 6
_NUM_GEMS = 4

# Per-pattern bonus and total patterns to complete in an episode.
_PATTERN_REWARD = 0.2
_TOTAL_PATTERNS = 4

# Match-progress reward share (the slice of the cumulative-1.0 cap that comes
# from raw gem-match counts; the remainder is allocated to pattern bonuses).
_MATCH_REWARD_CAP = 1.0 - _PATTERN_REWARD * _TOTAL_PATTERNS  # 0.2

# Total gems matched needed to saturate the match-progress reward. A standard
# 3-gem match pays 0.1 before this slice caps.
_TARGET_GEMS_MATCHED = 30

_GEM_SYMS = ("♦", "♣", "♠", "★")
# diamond, club, spade, star
_GEM_NAMES = ("diamond", "club", "spade", "star")
_SYM_EMPTY = "·"  # ·

# ---------------------------------------------------------------------------
# Action spec: swap-pair encoding (M1). Each action names a unique adjacent
# cell pair using either RIGHT (cell, cell to its right) or DOWN (cell, cell
# below). On a 6x6 board: 6*5 RIGHT + 5*6 DOWN = 60 swap actions <= 64.
# ---------------------------------------------------------------------------

_action_names: list[str] = []
_action_descs: list[str] = []
for _r in range(_SIZE):
    for _c in range(_SIZE - 1):
        _action_names.append(f"SWAP_R{_r}_C{_c}_RIGHT")
        _action_descs.append(
            f"swap gem at ({_r},{_c}) with the one to its right ({_r},{_c + 1})"
        )
for _r in range(_SIZE - 1):
    for _c in range(_SIZE):
        _action_names.append(f"SWAP_R{_r}_C{_c}_DOWN")
        _action_descs.append(
            f"swap gem at ({_r},{_c}) with the one below ({_r + 1},{_c})"
        )

MATCH3_ACTION_SPEC = ActionSpec(
    names=tuple(_action_names), descriptions=tuple(_action_descs)
)


def _decode_swap(action: int) -> tuple[int, int, int, int]:
    """Decode an action index into (r1, c1, r2, c2) cell pair."""
    n_right = _SIZE * (_SIZE - 1)
    if action < n_right:
        r = action // (_SIZE - 1)
        c = action % (_SIZE - 1)
        return r, c, r, c + 1
    rest = action - n_right
    r = rest // _SIZE
    c = rest % _SIZE
    return r, c, r + 1, c


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Match3Env(BaseGlyphEnv):
    """Match-3 gem-swapping on a 6x6 board with 4 gem types.

    Reward shape (M1):
      - Per-gem match progress contributes up to _MATCH_REWARD_CAP (= 0.2).
      - Each completed 2x2 target pattern fires +_PATTERN_REWARD; up to
        _TOTAL_PATTERNS = 4 patterns per episode.
      - Cumulative cap = 1.0.
    """

    action_spec = MATCH3_ACTION_SPEC
    noop_action_name = "SWAP_R0_C0_RIGHT"  # invalid swap acts as noop

    def __init__(self, max_turns: int = 60) -> None:
        super().__init__(max_turns=max_turns)
        # Board stores gem indices 0.._NUM_GEMS-1, or -1 for empty
        self._board: list[list[int]] = []
        self._score: int = 0
        self._total_matched: int = 0
        self._match_reward_emitted: float = 0.0
        self._no_moves: bool = False
        self._last_msg: str = ""
        # M1: target pattern is a 2x2 block of gem indices the agent must
        # produce somewhere on the board to claim a pattern bonus.
        self._target_pattern: list[list[int]] = [[0, 0], [0, 0]]
        self._patterns_found: int = 0

    def env_id(self) -> str:
        return "glyphbench/classics-match3-v0"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _fill_board(self) -> None:
        """Fill the board ensuring no initial matches."""
        for r in range(_SIZE):
            for c in range(_SIZE):
                while True:
                    gem = int(self.rng.integers(_NUM_GEMS))
                    self._board[r][c] = gem
                    # Check horizontal match
                    if (
                        c >= 2
                        and self._board[r][c - 1] == gem
                        and self._board[r][c - 2] == gem
                    ):
                        continue
                    # Check vertical match
                    if (
                        r >= 2
                        and self._board[r - 1][c] == gem
                        and self._board[r - 2][c] == gem
                    ):
                        continue
                    break

    def _roll_target_pattern(self) -> None:
        """Sample a fresh 2x2 target pattern. Require all four cells to be
        distinct gem types AND the pair (top-left, bottom-right) to differ
        from (top-right, bottom-left); this rules out simple 2-colour
        checkerboards and drives the per-board hit probability low."""
        for _ in range(200):
            pat = [
                [int(self.rng.integers(_NUM_GEMS)) for _ in range(2)]
                for _ in range(2)
            ]
            flat = {pat[0][0], pat[0][1], pat[1][0], pat[1][1]}
            if len(flat) == _NUM_GEMS:
                self._target_pattern = pat
                return
        # Fallback: a permutation of all gem types.
        self._target_pattern = [[0, 1], [2, 3]]

    def _board_contains_pattern(
        self, pattern: list[list[int]], anchor_cells: set[tuple[int, int]] | None = None
    ) -> bool:
        """Return True iff the 2x2 ``pattern`` appears anywhere on the board.

        If ``anchor_cells`` is provided, only count pattern occurrences whose
        2x2 block CONTAINS *every* anchor cell (used to detect a pattern that
        was *created by* the player's swap — the two swapped cells must both
        lie inside the matching block, otherwise the pattern was already
        present before the swap)."""
        for r in range(_SIZE - 1):
            for c in range(_SIZE - 1):
                if (
                    self._board[r][c] == pattern[0][0]
                    and self._board[r][c + 1] == pattern[0][1]
                    and self._board[r + 1][c] == pattern[1][0]
                    and self._board[r + 1][c + 1] == pattern[1][1]
                ):
                    if anchor_cells is None:
                        return True
                    block = {
                        (r, c),
                        (r, c + 1),
                        (r + 1, c),
                        (r + 1, c + 1),
                    }
                    if anchor_cells.issubset(block):
                        return True
        return False

    def _check_pattern_completion(
        self, anchor_cells: set[tuple[int, int]] | None = None
    ) -> float:
        """If the board currently contains the target pattern (optionally
        restricted to blocks intersecting ``anchor_cells``), mark it found
        and roll a fresh pattern. Returns the bonus reward earned this call.
        """
        if (
            self._patterns_found < _TOTAL_PATTERNS
            and self._board_contains_pattern(
                self._target_pattern, anchor_cells=anchor_cells
            )
        ):
            self._patterns_found += 1
            self._roll_target_pattern()
            return _PATTERN_REWARD
        return 0.0

    def _find_matches(self) -> set[tuple[int, int]]:
        """Find all cells involved in matches of 3+."""
        matched: set[tuple[int, int]] = set()
        # Horizontal
        for r in range(_SIZE):
            c = 0
            while c < _SIZE:
                gem = self._board[r][c]
                if gem < 0:
                    c += 1
                    continue
                run_len = 1
                while (
                    c + run_len < _SIZE
                    and self._board[r][c + run_len] == gem
                ):
                    run_len += 1
                if run_len >= 3:
                    for k in range(run_len):
                        matched.add((r, c + k))
                c += run_len
        # Vertical
        for c in range(_SIZE):
            r = 0
            while r < _SIZE:
                gem = self._board[r][c]
                if gem < 0:
                    r += 1
                    continue
                run_len = 1
                while (
                    r + run_len < _SIZE
                    and self._board[r + run_len][c] == gem
                ):
                    run_len += 1
                if run_len >= 3:
                    for k in range(run_len):
                        matched.add((r + k, c))
                r += run_len
        return matched

    def _remove_and_score(self, matched: set[tuple[int, int]]) -> int:
        """Remove matched cells. Returns count of matched gems."""
        count = len(matched)
        for r, c in matched:
            self._board[r][c] = -1
        return count

    def _gravity(self) -> None:
        """Drop gems down to fill empty spaces."""
        for c in range(_SIZE):
            write_row = _SIZE - 1
            for r in range(_SIZE - 1, -1, -1):
                if self._board[r][c] >= 0:
                    self._board[write_row][c] = self._board[r][c]
                    if write_row != r:
                        self._board[r][c] = -1
                    write_row -= 1
            # Fill top with new gems
            for r in range(write_row, -1, -1):
                self._board[r][c] = int(self.rng.integers(_NUM_GEMS))

    def _cascade(self) -> int:
        """Process all matches + gravity cascades. Returns the gem count for
        the FIRST (player-triggered) wave only. Subsequent cascades from
        falling gems still happen mechanically but contribute 0 to the
        per-step reward (in line with the prior contract)."""
        chain = 0
        first_wave_score = 0
        while True:
            matched = self._find_matches()
            if not matched:
                break
            chain += 1
            score = self._remove_and_score(matched)
            self._total_matched += len(matched)
            if chain == 1:
                first_wave_score = score
            self._gravity()
        return first_wave_score

    def _has_valid_move(self) -> bool:
        """Check if any swap-pair creates a match or target pattern."""
        for action in range(self.action_spec.n):
            r1, c1, r2, c2 = _decode_swap(action)
            self._board[r1][c1], self._board[r2][c2] = (
                self._board[r2][c2],
                self._board[r1][c1],
            )
            has_match = len(self._find_matches()) > 0
            has_pattern = (
                self._patterns_found < _TOTAL_PATTERNS
                and self._board_contains_pattern(
                    self._target_pattern,
                    anchor_cells={(r1, c1), (r2, c2)},
                )
            )
            self._board[r1][c1], self._board[r2][c2] = (
                self._board[r2][c2],
                self._board[r1][c1],
            )
            if has_match or has_pattern:
                return True
        return False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = [[-1] * _SIZE for _ in range(_SIZE)]
        self._score = 0
        self._total_matched = 0
        self._match_reward_emitted = 0.0
        self._patterns_found = 0
        self._no_moves = False
        self._last_msg = ""
        self._fill_board()
        if not self._has_valid_move():
            self._fill_board()  # retry
        # Roll initial target pattern AFTER the board is filled so we don't
        # accidentally ship a board that already contains the pattern.
        for _ in range(20):
            self._roll_target_pattern()
            if not self._board_contains_pattern(self._target_pattern):
                break
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}

        r1, c1, r2, c2 = _decode_swap(action)

        # Perform swap
        self._board[r1][c1], self._board[r2][c2] = (
            self._board[r2][c2],
            self._board[r1][c1],
        )

        anchor_cells = {(r1, c1), (r2, c2)}

        # M1: check whether the swap creates the target pattern at the
        # swapped cells. The pattern check happens BEFORE matches resolve so
        # the bonus depends only on the player's swap choice.
        pattern_bonus = self._check_pattern_completion(
            anchor_cells=anchor_cells
        )

        # Check whether the swap also creates a 3-in-a-row match.
        matched = self._find_matches()
        if not matched:
            if pattern_bonus > 0:
                # Pattern was completed — keep the swap in place to make the
                # mechanic legible (the agent can see their pattern reward).
                self._last_msg = (
                    f"Pattern completed! +{pattern_bonus:.2f}. Patterns: "
                    f"{self._patterns_found}/{_TOTAL_PATTERNS}"
                )
                if not self._has_valid_move():
                    self._no_moves = True
                    info["no_moves"] = True
                    self._last_msg += " No more valid moves. Game over."
                    return (
                        self._render_current_observation(),
                        pattern_bonus,
                        True,
                        False,
                        info,
                    )
                info["patterns_found"] = self._patterns_found
                return (
                    self._render_current_observation(),
                    pattern_bonus,
                    False,
                    False,
                    info,
                )
            # No match AND no pattern -> swap back, no reward.
            self._board[r1][c1], self._board[r2][c2] = (
                self._board[r2][c2],
                self._board[r1][c1],
            )
            self._last_msg = "Swap did not create a match. No effect."
            return (
                self._render_current_observation(),
                0.0,
                False,
                False,
                info,
            )

        # Process cascades for the player-triggered match.
        match_count = self._cascade()
        self._score += match_count

        # Match-progress reward, capped at _MATCH_REWARD_CAP cumulative.
        match_normalized = float(match_count) / _TARGET_GEMS_MATCHED
        match_remaining = max(
            0.0, _MATCH_REWARD_CAP - self._match_reward_emitted
        )
        match_reward = min(match_normalized, match_remaining)
        self._match_reward_emitted += match_reward

        reward = match_reward + pattern_bonus

        msg_parts = [f"Match! +{match_count} gems."]
        if pattern_bonus > 0:
            msg_parts.append(
                f"Target pattern completed! Patterns: "
                f"{self._patterns_found}/{_TOTAL_PATTERNS}"
            )
        self._last_msg = " ".join(msg_parts)

        # Check for game over (no valid moves remaining).
        if not self._has_valid_move():
            self._no_moves = True
            info["no_moves"] = True
            self._last_msg += " No more valid moves. Game over."
            return self._render_current_observation(), reward, True, False, info

        info["patterns_found"] = self._patterns_found
        info["total_matched"] = self._total_matched
        return self._render_current_observation(), reward, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_target_pattern(self) -> str:
        rows: list[str] = []
        for r in range(2):
            row_syms: list[str] = []
            for c in range(2):
                row_syms.append(_GEM_SYMS[self._target_pattern[r][c]])
            rows.append("".join(row_syms))
        return " | ".join(rows)

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_SIZE, _SIZE, fill=_SYM_EMPTY)
        syms: dict[str, str] = {}
        for r in range(_SIZE):
            for c in range(_SIZE):
                gem = self._board[r][c]
                if gem >= 0:
                    sym = _GEM_SYMS[gem]
                    grid[r][c] = sym
                    syms[sym] = _GEM_NAMES[gem]
                else:
                    syms[_SYM_EMPTY] = "empty (during cascade)"
        for row in self._target_pattern:
            for gem in row:
                syms[_GEM_SYMS[gem]] = _GEM_NAMES[gem]
        legend = build_legend(syms)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Total matched: {self._total_matched}    "
            f"Patterns: {self._patterns_found}/{_TOTAL_PATTERNS}\n"
            f"Target 2x2 pattern (row1 | row2): {self._render_target_pattern()}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._last_msg,
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- a Match-3 puzzle with target "
            "patterns.\n\n"
            "RULES\n"
            f"The board is {_SIZE}x{_SIZE} with {_NUM_GEMS} gem types.\n"
            "Each turn you swap two adjacent gems in a single action whose "
            "name encodes the cell pair (RIGHT or DOWN).\n"
            "The swap takes effect if it creates a match of 3+ in a "
            "row/column OR completes the target 2x2 pattern at the swapped "
            "cells; otherwise the swap is reverted.\n"
            "After matches, gems above fall down and new gems fill from the "
            "top.\n"
            "Cascades from falling gems continue the turn but do NOT multiply "
            "score.\n\n"
            "TARGET PATTERN (M1 permutation goal)\n"
            "The HUD shows a 2x2 TARGET PATTERN of gems. A pattern bonus fires "
            "only when the player's swap creates that exact block at the two "
            f"swapped cells, before matches and gravity resolve: +{_PATTERN_REWARD:.2f}; "
            "then a fresh target pattern is rolled.\n"
            f"Up to {_TOTAL_PATTERNS} pattern bonuses per episode.\n\n"
            "SCORING\n"
            f"  - Match progress contributes up to {_MATCH_REWARD_CAP:.2f} "
            f"cumulative (1/{_TARGET_GEMS_MATCHED} per gem matched, capped).\n"
            f"  - Each completed pattern fires +{_PATTERN_REWARD:.2f}.\n"
            f"  - Cumulative reward caps at 1.0.\n\n"
            f"ACTIONS\n"
            "Action names are SWAP_R<row>_C<col>_RIGHT (swap with the cell to "
            "the right) or SWAP_R<row>_C<col>_DOWN (swap with the cell "
            f"below). {self.action_spec.n} actions in total.\n"
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
