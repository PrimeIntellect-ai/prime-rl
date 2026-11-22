"""Atari Ms. Pac-Man environment.

A 28x31 maze. Player eats pellets and power pellets while avoiding ghosts.
Power pellets make ghosts frightened and edible.

Gym ID: glyphbench/atari-mspacman-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

# Direction vectors
_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# Classic maze template (28 wide x 31 tall)
# '█' = wall, '·' = pellet, '*' = power pellet, ' ' = empty corridor
_MAZE_TEMPLATE = [
    "████████████████████████████",
    "█·····················█····█",
    "█·████·█████·██·█████·████·█",
    "█*████·█████·██·█████·████*█",
    "█··························█",
    "█·████·██·████████·██·████·█",
    "█······██····██····██······█",
    "██████·█████ ██ █████·██████",
    "     █·██          ██·█     ",
    "     █·██ ███──███ ██·█     ",
    "██████·██ █      █ ██·██████",
    "      ·   █      █   ·      ",
    "██████·██ █      █ ██·██████",
    "     █·██ ████████ ██·█     ",
    "     █·██          ██·█     ",
    "██████·██ ████████ ██·██████",
    "█············██············█",
    "█·████·█████·██·█████·████·█",
    "█*··██·······  ·······██··*█",
    "███·██·██·████████·██·██·███",
    "█······██····██····██······█",
    "█·██████████·██·██████████·█",
    "█··························█",
    "████████████████████████████",
]

_MAZE_W = 28
_MAZE_H = len(_MAZE_TEMPLATE)

# Ghost spawn positions (inside the ghost pen)
_GHOST_SPAWNS = [(12, 11), (13, 11), (14, 11), (15, 11)]
_GHOST_CHARS = ["R", "P", "B", "O"]  # red, pink, blue, orange
_PLAYER_START = (14, 18)

_FRIGHTENED_DURATION = 20

# Pen interior + door: ghosts in this region are not yet "released"
_PEN_XS = range(11, 17)
_PEN_YS = range(10, 13)
_DOOR_CELLS = {(13, 9), (14, 9)}


def _in_pen_or_door(x: int, y: int) -> bool:
    if (x, y) in _DOOR_CELLS:
        return True
    return x in _PEN_XS and y in _PEN_YS

class MsPacManEnv(AtariBase):
    """Ms. Pac-Man: eat pellets, avoid ghosts, use power pellets.

    Actions: NOOP, UP, RIGHT, LEFT, DOWN
    Pattern D: +1/_WIN_TARGET per pellet/power-pellet eaten
    (full-scope = 213 dots; the actual count of pellets baked
    into the maze template). -1.0 if caught by a non-frightened
    ghost.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing this step",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
        ),
    )

    # Pattern D full-scope target: every pellet in the maze.
    # The default Ms.Pac-Man template has 209 regular + 4 power
    # pellets = 213. Each one yields +1/213.
    _WIN_TARGET: int = 213
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 1000) -> None:
        super().__init__(max_turns=max_turns)
        self._pellet_count: int = 0
        self._frightened_timer: int = 0
        self._ghost_eat_combo: int = 0
        self._player_dir: tuple[int, int] = (0, 0)
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-mspacman-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _task_description(self) -> str:
        return (
            "Eat all pellets (·) and power pellets (*) to clear the level. "
            "Avoid ghosts (R, P, B, O) or eat a power pellet to make them "
            "frightened (F) and edible. Eating ghosts gives bonus points."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Ms. Pac-Man.\n\n"
            "TASK\n"
            "Eat every pellet and power pellet in a classic Pac-Man "
            "maze without being caught by the four ghosts. Clearing "
            "all pellets advances the level.\n\n"
            "BOARD\n"
            "Fixed 28x24 maze. Walls '#', regular pellets '.' (one "
            "small dot), power pellets '*' (larger). Empty corridors "
            "are ' '. A horizontal tunnel wraps the maze (left edge "
            "connects to right edge). Ghost door 'dash' guards the "
            "ghost pen. Ghosts are 'R' (red), 'P' (pink), 'B' "
            "(blue), 'O' (orange); when frightened they show as 'F'. "
            "You are an arrow glyph starting at (14, 18).\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT move 1 cell if not blocked by "
            "a wall or the ghost door. Tunnel cells wrap. Ghosts "
            "each step pick the direction (excluding reverse) whose "
            "next cell is closest to you (or random when frightened "
            "or scattering). Eating a power pellet sets all ghosts "
            "to 'frightened' for 20 steps; during that time they "
            "move randomly and are edible.\n\n"
            "SCORING\n"
            "+1/213 reward per pellet (regular or power) eaten "
            "(Pattern D full-scope = 213 pellets, the maze total). "
            "Eating frightened ghosts yields no direct reward "
            "(only respawns them). -1.0 if a non-frightened ghost "
            "catches you.\n\n"
            "TERMINATION\n"
            "Being caught by a chase-mode ghost ends the episode "
            "with -1.0. Episode ends after all 213 pellets eaten "
            "(cumulative reward plateaus at +1.0) or after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, pellets remaining, power-up "
            "timer, and ghost states.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            "·": "pellet",
            "*": "power pellet",
            " ": "empty",
            "─": "ghost door",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(_MAZE_W, _MAZE_H)
        self._pellet_count = 0
        self._frightened_timer = 0
        self._ghost_eat_combo = 0
        self._player_dir = (0, 0)
        self._entities = []

        # Build maze from template
        for y, row in enumerate(_MAZE_TEMPLATE):
            for x, ch in enumerate(row):
                self._set_cell(x, y, ch)
                if ch in ("·", "*"):
                    self._pellet_count += 1

        # Place player
        self._player_x, self._player_y = _PLAYER_START

        # Spawn ghosts
        for i, (gx, gy) in enumerate(_GHOST_SPAWNS):
            ghost = self._add_entity(
                etype="ghost",
                char=_GHOST_CHARS[i],
                x=gx,
                y=gy,
            )
            ghost.data["home_x"] = gx
            ghost.data["home_y"] = gy
            ghost.data["color"] = _GHOST_CHARS[i]
            ghost.data["state"] = "scatter"  # chase / scatter / frightened
            ghost.data["dir"] = (0, -1)
            ghost.data["released"] = False

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Tick frightened timer
        if self._frightened_timer > 0:
            self._frightened_timer -= 1
            if self._frightened_timer == 0:
                self._ghost_eat_combo = 0
                for e in self._entities:
                    if e.etype == "ghost" and e.data.get("state") == "frightened":
                        e.data["state"] = "chase"
                        e.char = e.data["color"]

        # Move player
        if action_name in _DIRS:
            dx, dy = _DIRS[action_name]
            self._player_dir = (dx, dy)
        else:
            dx, dy = 0, 0

        nx, ny = self._player_x + dx, self._player_y + dy
        # Tunnel wrap
        if nx < 0:
            nx = _MAZE_W - 1
        elif nx >= _MAZE_W:
            nx = 0
        if not self._is_solid(nx, ny) and self._grid_at(nx, ny) != "─":
            self._player_x, self._player_y = nx, ny

        # Check pellet collection
        cell = self._grid_at(self._player_x, self._player_y)
        if cell == "·":
            self._set_cell(self._player_x, self._player_y, " ")
            self._on_point_scored(1)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._pellet_count -= 1
        elif cell == "*":
            self._set_cell(self._player_x, self._player_y, " ")
            self._on_point_scored(2)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._pellet_count -= 1
            # Frighten ghosts
            self._frightened_timer = _FRIGHTENED_DURATION
            self._ghost_eat_combo = 0
            for e in self._entities:
                if e.etype == "ghost" and e.alive:
                    e.data["state"] = "frightened"
                    e.char = "F"

        # Move ghosts
        for e in self._entities:
            if e.etype != "ghost" or not e.alive:
                continue
            self._move_ghost(e)

        # Check ghost collisions
        for e in self._entities:
            if e.etype != "ghost" or not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.data.get("state") == "frightened":
                    self._ghost_eat_combo += 1
                    # Respawn ghost (no direct reward)
                    e.x = e.data["home_x"]
                    e.y = e.data["home_y"]
                    e.data["state"] = "chase"
                    e.data["released"] = False
                    e.char = e.data["color"]
                else:
                    # Pattern D death penalty
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY

        # Win check (all pellets eaten)
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All pellets eaten!"

        info["pellets_remaining"] = self._pellet_count
        info["frightened_timer"] = self._frightened_timer
        return reward, self._game_over, info

    def _move_ghost(self, ghost: AtariEntity) -> None:
        """Simple ghost AI: chase player or move randomly.

        The ghost door is one-way: ghosts inside the pen can pass it to
        leave; once released, the door is treated as a wall so they
        cannot re-enter. Ghosts inside the pen always move randomly so
        chase-mode targeting cannot trap them against the bottom wall.
        """
        state = ghost.data.get("state", "chase")
        released = ghost.data.get("released", False)
        # Inside the pen, force random movement so chase-mode ghosts
        # don't pin themselves to the wall closest to the player.
        if not released and state == "chase":
            state = "scatter"

        def _passable(nx: int, ny: int) -> bool:
            # The door is "solid" in the base classifier, but ghosts in
            # the pen are allowed to cross it; once released, treat it
            # as a wall so they cannot re-enter.
            if self._grid_at(nx, ny) == "─":
                return not released
            return not self._is_solid(nx, ny)

        # Get possible directions (exclude reversing)
        cur_dir = ghost.data.get("dir", (0, -1))
        reverse = (-cur_dir[0], -cur_dir[1])
        possible: list[tuple[int, int]] = []
        for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if d == reverse:
                continue
            nx, ny = ghost.x + d[0], ghost.y + d[1]
            # Tunnel wrap
            if nx < 0:
                nx = _MAZE_W - 1
            elif nx >= _MAZE_W:
                nx = 0
            if _passable(nx, ny):
                possible.append(d)
        if not possible:
            # Allow reverse if stuck
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ghost.x + d[0], ghost.y + d[1]
                if nx < 0:
                    nx = _MAZE_W - 1
                elif nx >= _MAZE_W:
                    nx = 0
                if _passable(nx, ny):
                    possible.append(d)
        if not possible:
            return

        if state == "frightened":
            chosen = possible[int(self.rng.integers(0, len(possible)))]
        elif state == "chase":
            # Move toward player
            best_dist = float("inf")
            chosen = possible[0]
            for d in possible:
                nx, ny = ghost.x + d[0], ghost.y + d[1]
                dist = abs(nx - self._player_x) + abs(ny - self._player_y)
                if dist < best_dist:
                    best_dist = dist
                    chosen = d
        else:
            # scatter: random
            chosen = possible[int(self.rng.integers(0, len(possible)))]

        nx, ny = ghost.x + chosen[0], ghost.y + chosen[1]
        if nx < 0:
            nx = _MAZE_W - 1
        elif nx >= _MAZE_W:
            nx = 0
        ghost.x, ghost.y = nx, ny
        ghost.data["dir"] = chosen
        # Mark released once fully outside the pen+door region.
        if not released and not _in_pen_or_door(ghost.x, ghost.y):
            ghost.data["released"] = True

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        pwr = (
            str(self._frightened_timer)
            if self._frightened_timer > 0 else "OFF"
        )
        ghosts = []
        for e in self._entities:
            if e.etype == "ghost" and e.alive:
                st = e.data.get("state", "chase")
                ghosts.append(
                    f"{e.data['color']}={st}"
                )
        glist = ",".join(ghosts) if ghosts else "none"
        extra = (
            f"Pellets: {self._pellet_count}"
            f"  Power: {pwr}"
            f"  Ghosts: {glist}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: ghosts are moved in _game_step, skip default movement."""
        pass
