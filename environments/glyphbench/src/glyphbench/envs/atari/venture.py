"""Atari Venture environment.

Room-based dungeon crawler. Agent enters rooms, fights enemies,
collects treasures.

Gym ID: glyphbench/atari-venture-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class VentureEnv(AtariBase):
    """Venture: room-based dungeon crawler.

    The player navigates a hallway map and enters rooms containing
    enemies and treasures. Shoot enemies and collect treasures.

    Grid: 20 wide x 16 tall.
    Pattern D: +1/_WIN_TARGET per chamber treasure collected,
    -1 on contact with a monster.
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing",
            "shoot arrow in facing direction",
            "move up",
            "move right",
            "move left",
            "move down",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _NUM_ROOMS = 4

    # Pattern D full-scope: 4 chambers cleared (one treasure per
    # room). Each yields +1/4. Death by monster terminates with -1.
    _WIN_TARGET: int = 4
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._current_room: int = -1  # -1 = hallway
        self._facing: str = "RIGHT"
        self._collected: set[int] = set()
        self._arrow_cooldown: int = 0
        self._arrow_kill_reward: float = 0
        self._progress_count: int = 0
        self._death_pending: bool = False

    def env_id(self) -> str:
        return "glyphbench/atari-venture-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        self._death_pending = False
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._lives = 1
        self._current_room = -1
        self._facing = "RIGHT"
        self._collected = set()
        self._arrow_cooldown = 0
        self._arrow_kill_reward = 0
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._build_hallway()

    def _build_hallway(self) -> None:
        """Build the main hallway with room entrances."""
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []

        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "█")
            self._set_cell(x, self._HEIGHT - 1, "█")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "█")
            self._set_cell(self._WIDTH - 1, y, "█")

        # Room doors at cardinal positions
        door_positions = [
            (self._WIDTH // 2, 1),       # north room (0)
            (self._WIDTH - 2, self._HEIGHT // 2),  # east room (1)
            (self._WIDTH // 2, self._HEIGHT - 2),  # south room (2)
            (1, self._HEIGHT // 2),       # west room (3)
        ]
        for i, (dx, dy) in enumerate(door_positions):
            if i not in self._collected:
                self._add_entity("door", str(i + 1), dx, dy)

        # Hallway enemies
        self._add_entity("enemy", "E", 5, 5, dx=0, dy=1)
        self._add_entity("enemy", "E", 14, 10, dx=0, dy=-1)

    def _build_room(self, room_id: int) -> None:
        """Build interior of a specific room."""
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []

        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "█")
            self._set_cell(x, self._HEIGHT - 1, "█")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "█")
            self._set_cell(self._WIDTH - 1, y, "█")

        # Exit back to hallway
        self._set_cell(self._WIDTH // 2, self._HEIGHT - 1, " ")

        # Treasure in center
        if room_id not in self._collected:
            self._add_entity("treasure", "$", self._WIDTH // 2, self._HEIGHT // 2)

        # Enemies based on room
        n_enemies = 2 + room_id
        for i in range(n_enemies):
            ex = 3 + ((room_id * 7 + i * 5) % (self._WIDTH - 6))
            ey = 3 + ((room_id * 3 + i * 4) % (self._HEIGHT - 6))
            edx = 1 if i % 2 == 0 else -1
            edy = 0
            if i % 3 == 0:
                edx = 0
                edy = 1 if i % 4 == 0 else -1
            self._add_entity("enemy", "E", ex, ey, dx=edx, dy=edy)

        # Some internal walls
        wall_x = 5 + (room_id * 3) % 8
        for y in range(3, 7):
            self._set_cell(wall_x, y, "█")

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        self._arrow_kill_reward = 0
        info: dict[str, Any] = {}

        if self._arrow_cooldown > 0:
            self._arrow_cooldown -= 1

        dx, dy = 0, 0
        if action_name == "UP":
            dy = -1
            self._facing = "UP"
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            dy = 1
            self._facing = "DOWN"
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            dx = -1
            self._facing = "LEFT"
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            dx = 1
            self._facing = "RIGHT"
            self._player_dir = (1, 0)
        elif action_name == "FIRE":
            if self._arrow_cooldown <= 0:
                self._fire_arrow()
                self._arrow_cooldown = 3

        new_x = self._player_x + dx
        new_y = self._player_y + dy

        if (
            0 <= new_x < self._WIDTH
            and 0 <= new_y < self._HEIGHT
            and not self._is_solid(new_x, new_y)
        ):
            self._player_x = new_x
            self._player_y = new_y

        # Enemy bounce
        for e in self._entities:
            if e.etype == "enemy" and e.alive:
                nx = e.x + e.dx
                ny = e.y + e.dy
                if nx <= 1 or nx >= self._WIDTH - 2:
                    e.dx = -e.dx
                if ny <= 1 or ny >= self._HEIGHT - 2:
                    e.dy = -e.dy

        # Entity collision
        for e in self._entities:
            if not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.etype == "enemy":
                    # Single-life: monster contact ends the episode.
                    self._on_life_lost()
                    self._message = "Hit by enemy! Game Over."
                    reward = self._DEATH_PENALTY
                    return reward, self._game_over, info
                elif e.etype == "treasure":
                    e.alive = False
                    self._collected.add(self._current_room)
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Treasure!"
                elif e.etype == "door":
                    room_id = int(e.char) - 1
                    self._current_room = room_id
                    self._player_x = self._WIDTH // 2
                    self._player_y = self._HEIGHT - 2
                    self._build_room(room_id)
                    self._message = f"Entered room {room_id + 1}!"
                    break

        # Room exit: walking off bottom in a room
        if self._current_room >= 0 and self._player_y >= self._HEIGHT - 1:
            self._current_room = -1
            self._player_x = self._WIDTH // 2
            self._player_y = self._HEIGHT // 2
            self._build_hallway()
            self._message = "Back to hallway."

        # Win check (all 4 chambers cleared)
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Dungeon cleared!"

        info["current_room"] = self._current_room
        info["collected"] = len(self._collected)

        return reward, self._game_over, info

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Arrow kills are now reward-neutral; keep call signature.
        return obs, reward, terminated, truncated, info

    def _fire_arrow(self) -> None:
        """Fire arrow in facing direction."""
        adx, ady = 0, 0
        if self._facing == "UP":
            ady = -1
        elif self._facing == "DOWN":
            ady = 1
        elif self._facing == "LEFT":
            adx = -1
        elif self._facing == "RIGHT":
            adx = 1
        self._add_entity("arrow", "*", self._player_x + adx, self._player_y + ady, dx=adx, dy=ady)

    def _advance_entities(self) -> None:
        """Override to handle arrow-enemy collisions."""
        super()._advance_entities()
        # Check arrow hitting enemies (no reward; clearing only)
        arrows = [e for e in self._entities if e.etype == "arrow" and e.alive]
        enemies = [e for e in self._entities if e.etype == "enemy" and e.alive]
        for arrow in arrows:
            for enemy in enemies:
                if arrow.x == enemy.x and arrow.y == enemy.y:
                    arrow.alive = False
                    enemy.alive = False
                    self._message = "Enemy destroyed!"
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "empty",
            "$": "treasure",
            "E": "enemy",
            "*": "arrow",
            "D": "door",
            "1": "room 1 entrance",
            "2": "room 2 entrance",
            "3": "room 3 entrance",
            "4": "room 4 entrance",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        room = (
            self._current_room + 1
            if self._current_room >= 0
            else 0
        )
        extra = f"Room: {room}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Explore the dungeon. Enter rooms to find treasures. "
            "Shoot enemies with FIRE. Collect all treasures to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Venture.\n\n"
            "TASK\n"
            "Explore a hallway with 4 rooms. Enter each room, "
            "shoot enemies and grab the treasure inside. Collect "
            "all 4 treasures to advance to the next level.\n\n"
            "BOARD\n"
            "20x16. Walls '#'. From the hallway, enter rooms via "
            "numbered door tiles ('1', '2', '3', '4') at the 4 "
            "cardinal positions. Inside a room: treasure '$', "
            "enemies 'E' (walking), your arrow '*'. Exit a room by "
            "walking through its bottom row. You are an arrow "
            "glyph.\n\n"
            "MECHANICS\n"
            "UP/DOWN/LEFT/RIGHT move 1 cell and set facing. FIRE "
            "launches an arrow in facing direction (cooldown 3 "
            "steps). Stepping onto a door teleports you into the "
            "room. Enemies patrol with simple axis movement and "
            "bounce at walls. Each room also has an internal wall.\n\n"
            "SCORING\n"
            "Pattern D: +1/4 reward per chamber treasure "
            "collected. Killing enemies emits no reward (only "
            "clears the path). -1 reward on contact with a "
            "monster. Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any monster contact ends with -1. "
            "Collecting all 4 chamber treasures ends with "
            "cumulative +1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, facing, and current room "
            "number (0 = hallway, 1-4 = rooms).\n\n"
            + self.action_spec.render_for_prompt()
        )
