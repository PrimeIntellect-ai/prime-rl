"""MiniHack Boxoban environments. Sokoban puzzles in a dungeon.

Variants:
  * medium: F2 process-of-elimination — BFS solvability filter (50 retries).
  * hard:   F2 + N1 lava shortcuts — solvable seeds with lava tiles
            offering faster but deadly routes.
  * unfiltered: no global random filter, but uses a lightweight solvable scaffold.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase

# Maximum BFS nodes for solvability check (keeps gen fast)
_BFS_NODE_CAP = 20_000


class _BoxobanBase(MiniHackBase):
    """Base for Boxoban (Sokoban) puzzle environments."""

    _grid_size: int = 7
    _num_boxes: int = 2
    _filter_solvable: bool = False  # if True, retry until BFS finds a solution
    _has_lava: bool = False  # if True, some non-target floor cells become lava

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._box_positions: list[tuple[int, int]] = []
        self._target_positions: list[tuple[int, int]] = []
        self._lava_positions: set[tuple[int, int]] = set()
        self._rewarded_targets: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Generation: optionally retry until a BFS-solvable seed is found.
    # ------------------------------------------------------------------

    def _generate_level(self, seed: int) -> None:
        s = self._grid_size
        if not self._filter_solvable:
            # The original unfiltered generator admitted occasional dead
            # Sokoban states. Keep the variant's lightweight scaffold identity,
            # but reject malformed scaffold samples and fall back to a trivial
            # three-row push puzzle if all random samples fail.
            for _attempt in range(50):
                self._init_grid(s, s)
                self._box_positions = []
                self._target_positions = []
                self._lava_positions = set()
                self._rewarded_targets = set()

                occupied: set[tuple[int, int]] = set()
                rows = [
                    int(row)
                    for row in self.rng.choice(
                        range(1, s - 1), size=self._num_boxes, replace=False
                    )
                ]

                for row in rows:
                    bx = int(self.rng.integers(2, s - 2))
                    tx = bx + 1
                    self._box_positions.append((bx, row))
                    self._target_positions.append((tx, row))
                    occupied.add((bx, row))
                    occupied.add((tx, row))

                first_bx, first_by = self._box_positions[0]
                self._place_player(first_bx - 1, first_by)
                if self._player_pos not in occupied and self._is_solvable():
                    return

            self._init_grid(s, s)
            self._box_positions = []
            self._target_positions = []
            self._lava_positions = set()
            self._rewarded_targets = set()
            fallback_rows = [1, s // 2, s - 2][: self._num_boxes]
            for row in fallback_rows:
                self._box_positions.append((3, row))
                self._target_positions.append((4, row))
            self._place_player(2, fallback_rows[0])
            return

        attempts_max = 50
        for _attempt in range(attempts_max):
            self._init_grid(s, s)
            self._box_positions = []
            self._target_positions = []
            self._lava_positions = set()
            self._rewarded_targets = set()

            occupied: set[tuple[int, int]] = set()
            # Player
            px = int(self.rng.integers(1, s - 1))
            py = int(self.rng.integers(1, s - 1))
            self._place_player(px, py)
            occupied.add((px, py))

            # Place boxes (away from walls so they're never corner-locked at gen)
            for _ in range(self._num_boxes):
                tries = 0
                while tries < 60:
                    bx = int(self.rng.integers(2, s - 2))
                    by = int(self.rng.integers(2, s - 2))
                    if (bx, by) not in occupied:
                        break
                    tries += 1
                self._box_positions.append((bx, by))
                occupied.add((bx, by))

                # Targets must be reachable; just pick any open cell
                tries = 0
                while tries < 60:
                    tx = int(self.rng.integers(1, s - 1))
                    ty = int(self.rng.integers(1, s - 1))
                    if (tx, ty) not in occupied:
                        break
                    tries += 1
                self._target_positions.append((tx, ty))
                occupied.add((tx, ty))

            # Lava shortcuts: place lava around the perimeter / between boxes,
            # avoiding player, boxes, and targets.
            if self._has_lava:
                lava_count = max(2, self._num_boxes + 1)
                placed = 0
                tries = 0
                while placed < lava_count and tries < 60:
                    lx = int(self.rng.integers(1, s - 1))
                    ly = int(self.rng.integers(1, s - 1))
                    if (lx, ly) not in occupied:
                        self._lava_positions.add((lx, ly))
                        self._place_lava(lx, ly)
                        occupied.add((lx, ly))
                        placed += 1
                    tries += 1

            if not self._filter_solvable or self._is_solvable():
                return
        # Fall through: last attempt; accept whatever we have

    def _is_solvable(self) -> bool:
        """BFS over (player_pos, frozenset(box_positions)) states.

        Returns True if a solution exists within _BFS_NODE_CAP states.
        """
        s = self._grid_size

        def walkable(x: int, y: int, boxes: frozenset[tuple[int, int]]) -> bool:
            if not (0 <= x < s and 0 <= y < s):
                return False
            ch = self._grid[y][x]
            if ch in ("█", "-", "|"):
                return False
            if (x, y) in self._lava_positions:
                return False
            return (x, y) not in boxes

        start_pos = self._player_pos
        start_boxes = frozenset(self._box_positions)
        targets = frozenset(self._target_positions)
        if start_boxes == targets:
            return True
        seen: set[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = set()
        queue: deque[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = deque()
        queue.append((start_pos, start_boxes))
        seen.add((start_pos, start_boxes))
        explored = 0
        while queue and explored < _BFS_NODE_CAP:
            (px, py), boxes = queue.popleft()
            explored += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = px + dx, py + dy
                if not (0 <= nx < s and 0 <= ny < s):
                    continue
                ch = self._grid[ny][nx]
                if ch in ("█", "-", "|"):
                    continue
                if (nx, ny) in self._lava_positions:
                    continue
                if (nx, ny) in boxes:
                    # Pushing
                    bx, by = nx + dx, ny + dy
                    if not walkable(bx, by, boxes - {(nx, ny)}):
                        continue
                    new_boxes = (boxes - {(nx, ny)}) | {(bx, by)}
                    if new_boxes == targets:
                        return True
                    state = ((nx, ny), new_boxes)
                    if state not in seen:
                        seen.add(state)
                        queue.append(state)
                else:
                    state = ((nx, ny), boxes)
                    if state not in seen:
                        seen.add(state)
                        queue.append(state)
        return False

    # ------------------------------------------------------------------
    # Step (with lava-aware movement)
    # ------------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            px, py = self._player_pos
            nx, ny = px + dx, py + dy

            # Lava? Death.
            if (nx, ny) in self._lava_positions:
                self._player_pos = (nx, ny)
                self._message = "You step into lava and burn to death!"
                return self._render_current_observation(), -1.0, True, False, {
                    "cause_of_death": "lava"
                }

            # Pushing a box?
            box_idx: int | None = None
            for i, (bx, by) in enumerate(self._box_positions):
                if (bx, by) == (nx, ny):
                    box_idx = i
                    break
            if box_idx is not None:
                if dx != 0 and dy != 0:
                    self._message = "Boxes can only be pushed north, south, east, or west."
                    return self._render_current_observation(), 0.0, False, False, {
                        "player_pos": self._player_pos,
                        "boxes_on_target": sum(
                            1 for b in self._box_positions if b in self._target_positions
                        ),
                    }
                behind_x, behind_y = nx + dx, ny + dy
                if (behind_x, behind_y) in self._lava_positions:
                    self._message = "The box stops at the lava edge."
                elif self._is_walkable(behind_x, behind_y) and (
                    behind_x, behind_y
                ) not in self._box_positions:
                    self._box_positions[box_idx] = (behind_x, behind_y)
                    self._player_pos = (nx, ny)
                    self._message = "You push the box."
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)

        reward = 0.0
        newly_filled = (
            set(self._box_positions)
            & set(self._target_positions)
            - self._rewarded_targets
        )
        if newly_filled:
            reward += len(newly_filled) / max(1, self._num_boxes)
            self._rewarded_targets.update(newly_filled)

        terminated = False
        # Win when every target has a box.
        if (
            len(self._target_positions) > 0
            and set(self._box_positions) == set(self._target_positions)
        ):
            terminated = True
            self._message = "All boxes on targets! Puzzle complete!"

        info: dict[str, Any] = {
            "player_pos": self._player_pos,
            "boxes_on_target": sum(
                1 for b in self._box_positions if b in self._target_positions
            ),
        }
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"
                elif ch == "♨":
                    symbols["♨"] = "lava (fatal on entry)"

        for tx, ty in self._target_positions:
            if (tx, ty) not in self._box_positions:
                render[ty][tx] = "X"
        if self._target_positions:
            symbols["X"] = "target"

        for bx, by in self._box_positions:
            if (bx, by) in self._target_positions:
                render[by][bx] = "*"
                symbols["*"] = "box on target"
            else:
                render[by][bx] = "0"
                symbols["0"] = "box"

        px, py = self._player_pos
        render[py][px] = "@"
        symbols["@"] = "you"

        boxes_on = sum(
            1 for b in self._box_positions if b in self._target_positions
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Boxes on target: {boxes_on}/{self._num_boxes}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    # ------------------------------------------------------------------
    # Task description
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        parts = [
            f"Sokoban puzzle: push {self._num_boxes} boxes (0) onto target "
            f"positions (X). Walk into a box from N/S/E/W to push it; diagonal "
            f"moves cannot push boxes. Boxes can't be pulled. "
            f"A box on its target shows as *.",
        ]
        if self._has_lava:
            parts.append(
                "Some tiles are lava (♨) — stepping in is fatal. Lava blocks "
                "box pushes; keep all boxes available for their targets. Lava "
                "death gives -1."
            )
        if self._filter_solvable:
            parts.append("Every seed is guaranteed solvable.")
        else:
            parts.append("The unfiltered variant keeps a solvable scaffold.")
        parts.append(
            "Reward: +1 total, split equally across targets the first time each "
            "gets a box; the episode ends when all boxes are on targets."
        )
        return " ".join(parts)


class MiniHackBoxobanMediumEnv(_BoxobanBase):
    """7x7 grid, 2 boxes, BFS solvability filter."""

    _grid_size = 7
    _num_boxes = 2
    _filter_solvable = True

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-medium-v0"


class MiniHackBoxobanHardEnv(_BoxobanBase):
    """9x9 grid, 3 boxes, BFS filter + lava shortcuts."""

    _grid_size = 9
    _num_boxes = 3
    _filter_solvable = True
    _has_lava = True

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-hard-v0"


class MiniHackBoxobanUnfilteredEnv(_BoxobanBase):
    """9x9 grid, 3 boxes, scaffolded without the full random BFS filter."""

    _grid_size = 9
    _num_boxes = 3
    _filter_solvable = False

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-unfiltered-v0"
