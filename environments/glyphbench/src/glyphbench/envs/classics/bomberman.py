"""Bomberman: place bombs to destroy crates and reach the exit."""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIZE = 11
_BLAST_RADIUS = 2
_BOMB_TIMER = 3

_SYM_PLAYER = "@"
_SYM_WALL = "\u2588"    # █
_SYM_CRATE = "\u2593"   # ▓
_SYM_BOMB = "\u2297"     # ⊗
_SYM_EXPLOSION = "\u203b"  # ※
_SYM_EXIT = "\u2605"     # ★
_SYM_FLOOR = "\u00b7"    # ·

# Cell types
FLOOR = 0
WALL = 1
CRATE = 2
EXIT = 3

# ---------------------------------------------------------------------------
# Action spec
# ---------------------------------------------------------------------------

BOMBERMAN_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"),
    descriptions=(
        "move one cell up",
        "move one cell down",
        "move one cell left",
        "move one cell right",
        "place a bomb at your current position",
        "do nothing this turn",
    ),
)

_DIR_MAP = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BombermanEnv(BaseGlyphEnv):
    """Bomberman on an 11x11 grid."""

    action_spec = BOMBERMAN_ACTION_SPEC
    noop_action_name = "WAIT"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._grid: list[list[int]] = []
        self._player_r: int = 0
        self._player_c: int = 0
        # Bombs: list of (row, col, timer)
        self._bombs: list[list[int]] = []
        # Explosion cells this turn (for rendering)
        self._explosions: set[tuple[int, int]] = set()
        self._dead: bool = False
        self._won: bool = False
        self._crates_destroyed: int = 0
        self._total_crates: int = 1
        self._exit_r: int = 0
        self._exit_c: int = 0

    def env_id(self) -> str:
        return "glyphbench/classics-bomberman-v0"

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _generate_grid(self) -> None:
        """Generate the Bomberman grid."""
        self._grid = [[FLOOR] * _SIZE for _ in range(_SIZE)]

        # Border walls
        for i in range(_SIZE):
            self._grid[0][i] = WALL
            self._grid[_SIZE - 1][i] = WALL
            self._grid[i][0] = WALL
            self._grid[i][_SIZE - 1] = WALL

        # Interior pillar walls at even row/col (1-indexed interior = positions 2,4,6,8)
        for r in range(2, _SIZE - 1, 2):
            for c in range(2, _SIZE - 1, 2):
                self._grid[r][c] = WALL

        # Collect free cells (not wall, not player start area)
        # Player starts at (1,1). Keep (1,1), (1,2), (2,1) clear.
        safe_zone = {(1, 1), (1, 2), (2, 1)}
        free_cells: list[tuple[int, int]] = []
        for r in range(1, _SIZE - 1):
            for c in range(1, _SIZE - 1):
                if self._grid[r][c] == FLOOR and (r, c) not in safe_zone:
                    free_cells.append((r, c))

        # Place crates on ~40% of free cells
        self.rng.shuffle(free_cells)  # type: ignore[arg-type]
        num_crates = int(len(free_cells) * 0.4)
        num_crates = max(num_crates, 5)  # at least some crates
        crate_cells = free_cells[:num_crates]
        free_cells[num_crates:]

        for r, c in crate_cells:
            self._grid[r][c] = CRATE

        # Total crates -- used for [-1, 1] reward normalization.
        self._total_crates = max(1, len(crate_cells))

        # Place exit behind one of the crates
        exit_idx = int(self.rng.integers(len(crate_cells)))
        self._exit_r, self._exit_c = crate_cells[exit_idx]
        # The exit is hidden under the crate; revealed when crate is destroyed

    # ------------------------------------------------------------------
    # Bomb & explosion logic
    # ------------------------------------------------------------------

    def _tick_bombs(self) -> float:
        """Advance bomb timers, explode if timer reaches 0. Returns reward."""
        self._explosions.clear()
        reward = 0.0
        new_bombs: list[list[int]] = []

        for bomb in self._bombs:
            bomb[2] -= 1
            if bomb[2] <= 0:
                reward += self._explode(bomb[0], bomb[1])
            else:
                new_bombs.append(bomb)

        self._bombs = new_bombs
        return reward

    def _explode(self, br: int, bc: int) -> float:
        """Process explosion at (br, bc). Returns reward from crate destruction."""
        reward = 0.0
        self._explosions.add((br, bc))

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, _BLAST_RADIUS + 1):
                nr, nc = br + dr * dist, bc + dc * dist
                if nr < 0 or nr >= _SIZE or nc < 0 or nc >= _SIZE:
                    break
                if self._grid[nr][nc] == WALL:
                    break
                self._explosions.add((nr, nc))
                if self._grid[nr][nc] == CRATE:
                    self._grid[nr][nc] = FLOOR
                    self._crates_destroyed += 1
                    # Pattern D: half the [0, 1] cap is shared across crate
                    # destruction; the exit reward is the remaining +0.5.
                    reward += 0.5 / self._total_crates
                    # If this crate was hiding the exit, reveal it
                    if nr == self._exit_r and nc == self._exit_c:
                        self._grid[nr][nc] = EXIT
                    break  # blast stops at crate

        # Check if player is caught in explosion
        if (self._player_r, self._player_c) in self._explosions:
            self._dead = True

        return reward

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._dead = False
        self._won = False
        self._bombs = []
        self._explosions = set()
        self._crates_destroyed = 0
        self._player_r = 1
        self._player_c = 1
        self._generate_grid()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        reward = 0.0

        if self._dead or self._won:
            return self._render_current_observation(), 0.0, True, False, info

        # Process action
        if name in _DIR_MAP:
            dr, dc = _DIR_MAP[name]
            nr, nc = self._player_r + dr, self._player_c + dc
            if 0 <= nr < _SIZE and 0 <= nc < _SIZE:
                cell = self._grid[nr][nc]
                if cell in (FLOOR, EXIT):
                    # Also check no bomb at destination (player can walk over bombs)
                    self._player_r = nr
                    self._player_c = nc
        elif name == "BOMB":
            # Place bomb at player position if not already a bomb there
            already = any(b[0] == self._player_r and b[1] == self._player_c for b in self._bombs)
            if not already:
                self._bombs.append([self._player_r, self._player_c, _BOMB_TIMER])

        # Tick bombs
        bomb_reward = self._tick_bombs()
        reward += bomb_reward

        # Check death from explosion
        if self._dead:
            info["dead"] = True
            return self._render_current_observation(), reward - 1.0, True, False, info

        # Check if player is on exit. Pattern D: terminal +0.5 (combined
        # with the crate cap of +0.5 keeps cumulative <= 1.0).
        if self._grid[self._player_r][self._player_c] == EXIT:
            self._won = True
            info["win"] = True
            return self._render_current_observation(), reward + 0.5, True, False, info

        return self._render_current_observation(), reward, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_SIZE, _SIZE, fill=_SYM_FLOOR)
        syms: dict[str, str] = {
            _SYM_FLOOR: "floor",
            _SYM_PLAYER: "you (player)",
            _SYM_WALL: "indestructible wall",
        }

        for r in range(_SIZE):
            for c in range(_SIZE):
                cell = self._grid[r][c]
                if cell == WALL:
                    grid[r][c] = _SYM_WALL
                elif cell == CRATE:
                    grid[r][c] = _SYM_CRATE
                    syms[_SYM_CRATE] = "destructible crate"
                elif cell == EXIT:
                    grid[r][c] = _SYM_EXIT
                    syms[_SYM_EXIT] = "exit (reach to win)"

        # Draw bombs
        for bomb in self._bombs:
            grid[bomb[0]][bomb[1]] = _SYM_BOMB
            syms[_SYM_BOMB] = f"bomb (explodes in {bomb[2]} turns)"

        # Draw explosions
        for r, c in self._explosions:
            grid[r][c] = _SYM_EXPLOSION
            syms[_SYM_EXPLOSION] = "explosion (deadly)"

        # Draw player on top (unless dead)
        if not self._dead:
            grid[self._player_r][self._player_c] = _SYM_PLAYER

        legend = build_legend(syms)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Crates destroyed: {self._crates_destroyed}    "
            f"Bombs active: {len(self._bombs)}"
        )
        msg = ""
        if self._dead:
            msg = "You were caught in an explosion! Game over."
        elif self._won:
            msg = "You reached the exit! You win!"

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- Bomberman.\n\n"
            "RULES\n"
            f"The grid is {_SIZE}x{_SIZE}.\n"
            "Indestructible walls form a grid pattern. Destructible crates fill "
            "~40% of open cells. An exit is hidden behind one crate.\n\n"
            "BOMBS\n"
            f"Place a bomb with BOMB. It explodes after {_BOMB_TIMER} turns with "
            f"blast radius {_BLAST_RADIUS} in all 4 cardinal directions.\n"
            "Explosions destroy crates (revealing what is behind them) and kill "
            "you if you are in the blast zone.\n\n"
            "GOAL\n"
            "Destroy crates to find and reach the exit.\n"
            "  Reward is +0.5/N for each of the N crates destroyed and\n"
            "  +0.5 terminal for reaching the exit. Total max = 1.0.\n"
            "  -1 reward if caught in an explosion (terminal).\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
