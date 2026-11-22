"""Atari Centipede environment.

Centipede winds through mushrooms; player moves in bottom area.

Gym ID: glyphbench/atari-centipede-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class CentipedeEnv(AtariBase):
    """Centipede: shoot the centipede winding through mushrooms.

    20x20 grid. Centipede descends through a mushroom field.
    Player moves in the bottom 4 rows.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    Pattern A: +1/_WIN_TARGET per kill (segment, mushroom, or spider).
    No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up (within bottom zone)",
            "move down",
            "move left",
            "move right",
            "fire a bullet upward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 18
    _PLAYER_ZONE_TOP = 16

    # Pattern A full-scope target: 30 kills (segments + mushrooms +
    # spiders, all counting equally toward the bound).
    _WIN_TARGET: int = 30

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._segments: list[AtariEntity] = []
        self._mushrooms: set[tuple[int, int]] = set()
        self._bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spider: AtariEntity | None = None
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-centipede-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._segments = []
        self._bullets = []
        self._step_counter = 0
        self._spider = None

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Scatter mushrooms
        self._mushrooms = set()
        rng = self.rng
        n_mush = 15 + self._level * 3
        for _ in range(min(n_mush, 40)):
            mx = int(rng.integers(2, self._WIDTH - 2))
            my = int(rng.integers(2, self._PLAYER_ZONE_TOP))
            self._mushrooms.add((mx, my))

        # Create centipede (chain of segments)
        length = min(8 + self._level * 2, 14)
        for i in range(length):
            seg = self._add_entity(
                "segment", "O" if i == 0 else "o",
                self._WIDTH - 2 - i, 1,
            )
            seg.data["dir"] = -1
            seg.data["idx"] = i
            self._segments.append(seg)

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Player movement (bottom 4 rows)
        if (
            action_name == "UP"
            and self._player_y > self._PLAYER_ZONE_TOP
        ):
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif (
            action_name == "DOWN"
            and self._player_y < self._HEIGHT - 2
        ):
            self._player_y += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "FIRE" and len(self._bullets) < 1:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1,
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False
                continue
            # Hit mushroom
            if (b.x, b.y) in self._mushrooms:
                self._mushrooms.discard((b.x, b.y))
                b.alive = False
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                continue
            # Hit segment
            for seg in self._segments:
                if (
                    seg.alive
                    and seg.x == b.x
                    and seg.y == b.y
                ):
                    seg.alive = False
                    b.alive = False
                    self._on_point_scored(2)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    # Leave mushroom where segment died
                    self._mushrooms.add((seg.x, seg.y))
                    self._message = "Segment hit!"
                    break
            # Hit spider
            if (
                b.alive
                and self._spider
                and self._spider.alive
                and self._spider.x == b.x
                and self._spider.y == b.y
            ):
                self._spider.alive = False
                b.alive = False
                self._on_point_scored(3)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Spider killed!"

        self._bullets = [b for b in self._bullets if b.alive]

        # Move centipede segments
        if self._step_counter % 2 == 0:
            for seg in self._segments:
                if not seg.alive:
                    continue
                d = seg.data.get("dir", 1)
                nx = seg.x + d
                # Check if hit wall or mushroom
                hit_obstacle = (
                    nx <= 0
                    or nx >= self._WIDTH - 1
                    or (nx, seg.y) in self._mushrooms
                )
                if hit_obstacle:
                    seg.data["dir"] = -d
                    seg.y += 1
                    if seg.y >= self._HEIGHT - 1:
                        seg.y = self._HEIGHT - 2
                else:
                    seg.x = nx

                # Check collision with player
                if (
                    seg.x == self._player_x
                    and seg.y == self._player_y
                ):
                    self._on_life_lost()
                    self._message = "Centipede got you!"
                    self._player_x = self._WIDTH // 2
                    self._player_y = self._PLAYER_Y

        self._segments = [
            s for s in self._segments if s.alive
        ]

        # Spider spawning and movement
        if (
            self._spider is None or not self._spider.alive
        ) and self._step_counter % 20 == 0:
            sy = int(
                self.rng.integers(
                    self._PLAYER_ZONE_TOP, self._HEIGHT - 2,
                )
            )
            self._spider = self._add_entity(
                "spider", "X", 1, sy, dx=1,
            )
            self._spider.data["dy"] = 1

        if self._spider and self._spider.alive and self._step_counter % 2 == 0:
            self._spider.x += self._spider.dx
            sdy = self._spider.data.get("dy", 1)
            self._spider.y += sdy
            if (
                self._spider.y <= self._PLAYER_ZONE_TOP
                or self._spider.y >= self._HEIGHT - 2
            ):
                self._spider.data["dy"] = -sdy
            if (
                self._spider.x <= 0
                or self._spider.x >= self._WIDTH - 1
            ):
                self._spider.alive = False
            # Eat mushrooms
            pos = (self._spider.x, self._spider.y)
            if pos in self._mushrooms:
                self._mushrooms.discard(pos)
            # Hit player
            if (
                self._spider.x == self._player_x
                and self._spider.y == self._player_y
            ):
                self._spider.alive = False
                self._on_life_lost()
                self._message = "Spider got you!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._PLAYER_Y

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "Target reached!"

        # Level clear
        if not self._segments and not self._game_over:
            self._level += 1
            self._message = "Centipede destroyed!"
            self._generate_level(self._level)

        self._redraw()
        info["segments"] = len(self._segments)
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for mx, my in self._mushrooms:
            self._set_cell(mx, my, "M")
        for seg in self._segments:
            if seg.alive:
                self._set_cell(seg.x, seg.y, seg.char)
        if self._spider and self._spider.alive:
            self._set_cell(
                self._spider.x, self._spider.y, "X"
            )
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "O": "centipede head", "o": "centipede body",
            "M": "mushroom", "X": "spider",
            "!": "your bullet", " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        segs = sum(1 for s in self._segments if s.alive)
        mush = len(self._mushrooms)
        extra = (
            f"Segments: {segs}  Mushrooms: {mush}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Shoot the centipede as it winds through mushrooms. "
            "Move in the bottom 4 rows. Avoid the spider. "
            "Destroyed segments leave mushrooms."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Centipede.\n\n"
            "TASK\n"
            "Shoot every segment of a centipede winding through a "
            "mushroom field. Clearing the centipede advances the "
            "level and spawns a longer one.\n\n"
            "BOARD\n"
            "20x20 field with wall borders. Mushrooms 'M' are scattered "
            "in the upper area. Centipede is a chain of cells: head 'O' "
            "and body 'o'. A spider 'X' may appear in your zone. "
            "Your bullets are '!'. You are an arrow glyph confined to "
            "a band of rows along the bottom (the [Grid] shows the bounds).\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT move you 1 cell (UP only inside "
            "the player zone). FIRE launches a single bullet upward "
            "(only 1 bullet alive at a time). Centipede segments each "
            "move every 2 steps: they travel horizontally; when they "
            "hit a wall or mushroom they reverse direction and step "
            "one row down. A shot segment leaves a mushroom at its "
            "death cell. Spider spawns every 20 steps, ping-pongs "
            "in the player zone, and eats mushrooms it touches.\n\n"
            "SCORING\n"
            "+1 progress unit per kill (centipede segment, "
            "mushroom, or spider), regardless of type. No "
            "per-step penalty.\n\n"
            "TERMINATION\n"
            "Single-life: centipede-player contact or "
            "spider-player contact ends the episode. The episode "
            "also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, segments remaining, and "
            "mushroom count.\n\n"
            + self.action_spec.render_for_prompt()
        )
