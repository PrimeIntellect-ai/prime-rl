"""Sokoban: push boxes onto targets.

Gym IDs:
  glyphbench/classics-sokoban-easy-v0    (7x7, 1-2 boxes)
  glyphbench/classics-sokoban-medium-v0  (9x9, 3 boxes)
  glyphbench/classics-sokoban-hard-v0    (11x11, 4-5 boxes)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOKOBAN_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "move/push up",
        "move/push down",
        "move/push left",
        "move/push right",
    ),
)

SYM_PLAYER = "@"
SYM_BOX = "\u25fc"       # ◼
SYM_TARGET = "\u25ce"     # ◎
SYM_BOX_ON_TARGET = "\u25c8"  # ◈
SYM_WALL = "\u2588"       # █
SYM_FLOOR = "\u00b7"      # ·

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# ---------------------------------------------------------------------------
# Base Sokoban Env
# ---------------------------------------------------------------------------


class _SokobanBase(BaseGlyphEnv):
    """Sokoban: push all boxes onto target positions."""

    action_spec = SOKOBAN_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 7
    _num_boxes: int = 2
    _difficulty: str = "easy"
    # Phase 3 audit fix: minimum reverse-scramble steps applied to EACH box during
    # generation. Larger _PULL_DEPTH = boxes start farther from their targets
    # = fewer free pre-solved components.
    _PULL_DEPTH: int = 3
    # Generation rejects puzzles where any box ends up on a target.
    _GEN_RETRIES: int = 50
    # Phase 3.5 fix: lowest depth we tolerate when retrying after a failed
    # generation pass. The off-target invariant is non-negotiable; we'd
    # rather ship an easier puzzle than one with pre-solved components.
    _MIN_PULL_DEPTH: int = 3

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._walls: set[tuple[int, int]] = set()
        self._boxes: set[tuple[int, int]] = set()
        self._targets: set[tuple[int, int]] = set()
        self._player: tuple[int, int] = (0, 0)
        self._boxes_on_targets: int = 0
        self._solution_pushes: list[tuple[tuple[int, int], int, int]] = []

    def env_id(self) -> str:
        return f"glyphbench/classics-sokoban-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Puzzle generation
    # ------------------------------------------------------------------

    def _generate_puzzle(self) -> None:
        """Generate a solvable Sokoban puzzle via reverse placement.

        Phase 3.5 fix: the off-target invariant is non-negotiable. We try at
        full ``_PULL_DEPTH`` first; on exhaustion of ``_GEN_RETRIES`` we
        decay the depth (down to ``_MIN_PULL_DEPTH``) and try again. The
        fallback NEVER ships a state with a box on a target.
        """
        depth = self._PULL_DEPTH
        while depth >= self._MIN_PULL_DEPTH:
            if self._try_generate_at_depth(depth):
                return
            depth -= 1
        # Last-resort: depth at the minimum, accept any solvable puzzle that
        # still satisfies the off-target invariant. If we somehow can't even
        # get that, surface it loudly so we notice in tests.
        if not self._try_generate_at_depth(
            self._MIN_PULL_DEPTH, retries=self._GEN_RETRIES * 4
        ):
            raise RuntimeError(
                f"Sokoban {self._difficulty} generator could not produce a "
                f"puzzle with no box on target after extensive retries"
            )

    def _try_generate_at_depth(
        self, depth: int, retries: int | None = None
    ) -> bool:
        """Single generation pass at the given pull depth. Returns True iff
        the resulting puzzle has every box off-target. Mutates env state on
        success."""
        s = self._grid_size
        n = self._num_boxes
        gen_retries = self._GEN_RETRIES if retries is None else retries

        for _ in range(gen_retries):
            self._solution_pushes = []
            self._walls = set()
            for x in range(s):
                self._walls.add((x, 0))
                self._walls.add((x, s - 1))
            for y in range(s):
                self._walls.add((0, y))
                self._walls.add((s - 1, y))

            interior = [
                (x, y)
                for x in range(1, s - 1)
                for y in range(1, s - 1)
            ]
            num_interior_walls = int(
                self.rng.integers(0, max(1, (s * s) // 20))
            )
            shuffled = list(interior)
            self.rng.shuffle(shuffled)

            # Reserve cells for targets first; remaining used for placement.
            cursor = 0
            targets_list = shuffled[cursor : cursor + n]
            self._targets = set(targets_list)
            cursor += n

            # Optional interior walls (must not collide with targets).
            wall_picks = []
            while (
                len(wall_picks) < num_interior_walls
                and cursor < len(shuffled)
            ):
                cell = shuffled[cursor]
                cursor += 1
                if cell in self._targets:
                    continue
                wall_picks.append(cell)
                self._walls.add(cell)

            # Initial state: boxes on targets, then reverse-scramble each box
            # _PULL_DEPTH times. Each reverse step places the player where the
            # corresponding forward push can undo it.
            self._boxes = set(targets_list)
            occupied: set[tuple[int, int]] = self._walls | self._boxes

            # Pick a starting player position adjacent to one of the targets.
            player_seed: tuple[int, int] | None = None
            target_iter = list(targets_list)
            self.rng.shuffle(target_iter)
            for tx, ty in target_iter:
                for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    cx, cy = tx - ddx, ty - ddy
                    if (
                        1 <= cx < s - 1
                        and 1 <= cy < s - 1
                        and (cx, cy) not in occupied
                    ):
                        player_seed = (cx, cy)
                        break
                if player_seed:
                    break
            if player_seed is None:
                continue
            self._player = player_seed

            ok = True
            for target in targets_list:
                if not self._pull_box(target, depth, s):
                    ok = False
                    break
            if not ok:
                continue

            # Reject puzzles where any box ends up back on a target.
            if self._boxes & self._targets:
                continue

            # Pick the player from any final-state cell that can reach the
            # generated solution sequence's first push position.
            free_cells = [
                (x, y)
                for x in range(1, s - 1)
                for y in range(1, s - 1)
                if (x, y) not in self._walls
                and (x, y) not in self._boxes
            ]
            if not free_cells:
                continue
            blocked = self._walls | self._boxes
            reachable = self._reachable_cells(self._player, blocked, s)
            candidates = [
                c
                for c in reachable
                if c not in self._boxes and c not in self._targets
            ]
            if not candidates:
                candidates = list(reachable) or free_cells
            idx = int(self.rng.integers(0, len(candidates)))
            self._player = candidates[idx]

            self._boxes_on_targets = len(self._boxes & self._targets)
            if not self._has_constructed_solution(s):
                continue
            return True

        # Could not produce a valid puzzle at this depth.
        return False

    def _pull_box(
        self,
        start: tuple[int, int],
        depth: int,
        grid_size: int,
    ) -> bool:
        """Reverse-scramble a box from ``start`` for ``depth`` steps.

        Each step moves the box one cell away from its current position and
        places the player on the far side, where a legal forward push can move
        that box back. A solver check after generation verifies the full puzzle
        remains forward-solvable.
        """
        bx, by = start
        # Box must currently be at start.
        if (bx, by) not in self._boxes:
            return False
        s = grid_size
        for _ in range(depth):
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            order = list(range(4))
            self.rng.shuffle(order)
            moved = False
            for di in order:
                ddx, ddy = dirs[di]
                old_player = self._player
                new_bx, new_by = bx + ddx, by + ddy
                push_from = (bx + 2 * ddx, by + 2 * ddy)
                if not (
                    1 <= new_bx < s - 1
                    and 1 <= new_by < s - 1
                    and 1 <= push_from[0] < s - 1
                    and 1 <= push_from[1] < s - 1
                ):
                    continue
                if (new_bx, new_by) in self._walls or (
                    new_bx,
                    new_by,
                ) in self._boxes:
                    continue
                # Reject moving onto a target — we want every box off-target
                # at generation end.
                if (new_bx, new_by) in self._targets:
                    continue
                # The player must be able to stand where the inverse forward
                # push starts; walls/boxes block, but targets don't.
                if push_from in self._walls or push_from in self._boxes:
                    continue
                # After the inverse push, the player stands at the box's
                # scrambled position. They must be able to walk to the
                # previous solution-sequence player position.
                blocked_before_move = self._walls | self._boxes
                if old_player not in self._reachable_cells(
                    (new_bx, new_by), blocked_before_move, s
                ):
                    continue
                # Apply reverse step: move box away and place player where
                # pushing it back is legal.
                self._boxes.discard((bx, by))
                self._boxes.add((new_bx, new_by))
                self._player = push_from
                self._solution_pushes.append((push_from, -ddx, -ddy))
                bx, by = new_bx, new_by
                moved = True
                break
            if not moved:
                return False
        return True

    def _has_constructed_solution(self, grid_size: int) -> bool:
        """Verify the recorded reverse-scramble pushes solve this puzzle."""
        player = self._player
        boxes = set(self._boxes)
        for push_from, dx, dy in reversed(self._solution_pushes):
            if push_from not in self._reachable_cells(
                player, self._walls | boxes, grid_size
            ):
                return False
            box_pos = (push_from[0] + dx, push_from[1] + dy)
            box_dest = (box_pos[0] + dx, box_pos[1] + dy)
            if box_pos not in boxes:
                return False
            if box_dest in self._walls or box_dest in (boxes - {box_pos}):
                return False
            boxes.remove(box_pos)
            boxes.add(box_dest)
            player = box_pos
        return boxes == self._targets

    def _is_solvable(self, state_cap: int = 300_000) -> bool:
        """Forward BFS over Sokoban states to reject unsolvable generations."""
        targets = frozenset(self._targets)
        start = (self._player, frozenset(self._boxes))
        if start[1] == targets:
            return True

        queue: deque[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = deque([start])
        seen = {start}
        while queue and len(seen) < state_cap:
            (px, py), boxes = queue.popleft()
            for dx, dy in _DIR_DELTAS.values():
                nx, ny = px + dx, py + dy
                if (nx, ny) in self._walls:
                    continue
                if (nx, ny) in boxes:
                    bx, by = nx + dx, ny + dy
                    if (bx, by) in self._walls or (bx, by) in boxes:
                        continue
                    next_boxes = frozenset((boxes - {(nx, ny)}) | {(bx, by)})
                    if next_boxes == targets:
                        return True
                    next_state = ((nx, ny), next_boxes)
                else:
                    next_state = ((nx, ny), boxes)
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append(next_state)
        return False

    @staticmethod
    def _reachable_cells(
        start: tuple[int, int],
        blocked: set[tuple[int, int]],
        grid_size: int,
    ) -> set[tuple[int, int]]:
        """BFS to find all reachable cells from start."""
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (1 <= nx < grid_size - 1 and 1 <= ny < grid_size - 1
                        and (nx, ny) not in blocked and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_puzzle()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        dx, dy = _DIR_DELTAS[name]
        px, py = self._player
        nx, ny = px + dx, py + dy

        old_on_targets = len(self._boxes & self._targets)

        # Check if target cell is free
        if (nx, ny) in self._walls:
            # Bump into wall -- no move
            pass
        elif (nx, ny) in self._boxes:
            # Pushing a box
            bx, by = nx + dx, ny + dy
            if (bx, by) not in self._walls and (bx, by) not in self._boxes:
                self._boxes.discard((nx, ny))
                self._boxes.add((bx, by))
                self._player = (nx, ny)
        else:
            self._player = (nx, ny)

        new_on_targets = len(self._boxes & self._targets)
        # Pattern A: each box placed/unplaced changes reward by +/-1/N so
        # cumulative reward telescopes to placed_final / N. Solved = 1.0.
        # No completion bonus -- we already cap at 1.0 by construction.
        n_targets = max(1, len(self._targets))
        reward = float(new_on_targets - old_on_targets) / n_targets
        self._boxes_on_targets = new_on_targets

        terminated = new_on_targets == len(self._targets)

        info["boxes_on_targets"] = new_on_targets
        info["total_targets"] = len(self._targets)
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_FLOOR)

        for wx, wy in self._walls:
            grid[wy][wx] = SYM_WALL

        for tx, ty in self._targets:
            if (tx, ty) not in self._boxes:
                grid[ty][tx] = SYM_TARGET

        for bx, by in self._boxes:
            if (bx, by) in self._targets:
                grid[by][bx] = SYM_BOX_ON_TARGET
            else:
                grid[by][bx] = SYM_BOX

        px, py = self._player
        grid[py][px] = SYM_PLAYER
        standing_on = "target" if self._player in self._targets else "floor"

        legend = build_legend({
            SYM_PLAYER: "player (you)",
            SYM_BOX: "box",
            SYM_TARGET: "target",
            SYM_BOX_ON_TARGET: "box on target",
            SYM_WALL: "wall",
            SYM_FLOOR: "floor",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Boxes on targets: {self._boxes_on_targets} / {len(self._targets)}    "
            f"Standing on: {standing_on}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Push all boxes onto target positions. You can push a box by walking "
            "into it, but only if the space behind the box is free. You cannot pull boxes.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size}.\n"
            f"- There are {self._num_boxes} boxes and {self._num_boxes} targets.\n"
            "- Moving into a box pushes it one cell in the same direction, if the cell behind it is empty.\n"
            f"- You get +{1.0 / self._num_boxes:.4f} reward when a box lands on a target, the\n"
            f"  same magnitude negative when a box is pushed off a target.\n"
            "- All boxes on targets ends the episode with cumulative reward = 1.0.\n"
            "- The player glyph (@) can cover a target; use the HUD's Standing on field.\n"
            "- Be careful: pushing a box into a corner may make the puzzle unsolvable.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class SokobanEasyEnv(_SokobanBase):
    _grid_size = 7
    _num_boxes = 2
    _difficulty = "easy"
    _PULL_DEPTH = 3

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


class SokobanMediumEnv(_SokobanBase):
    _grid_size = 9
    _num_boxes = 3
    _difficulty = "medium"
    _PULL_DEPTH = 5

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


class SokobanHardEnv(_SokobanBase):
    _grid_size = 11
    _num_boxes = 5
    _difficulty = "hard"
    _PULL_DEPTH = 7
    # Phase 3.5 fix: hard variant needs more attempts at full depth — the
    # reverse-pull search frequently dead-ends with 5 boxes on an 11x11
    # board, and the previous 50-retry budget hit the fallback path on ~12%
    # of seeds, occasionally shipping pre-solved components.
    _GEN_RETRIES = 200

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
