"""Atari Yars' Revenge environment.

Asymmetric shooter: eat the shield, fire the cannon at the Qotile.

Gym ID: glyphbench/atari-yarsrevenge-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class YarsRevengeEnv(AtariBase):
    """Yars' Revenge: eat shield cells and destroy the Qotile.

    20x20 grid. Qotile on right behind a shield wall.
    Eat shield cells to create a gap, then fire the Zorlon
    cannon to destroy the Qotile. Dodge the Destroyer missile.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    Pattern D: +1/_WIN_TARGET per Qotile kill (each kill is one
    cycle of shield-eating + Qotile-shooting). Eating shield
    cells emits no reward (only energy). -1 on Destroyer contact.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "fire Zorlon cannon (needs energy)",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _SHIELD_X = 14
    _QOTILE_X = 17
    _NEUTRAL_LEFT = 8
    _NEUTRAL_RIGHT = 10
    _MAX_ENERGY = 6

    # Pattern D full-scope: 3 Qotile kill cycles. Each kill yields
    # +1/3. Destroyer contact ends the episode with -1.
    _WIN_TARGET: int = 3
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._shield_cells: set[tuple[int, int]] = set()
        self._energy: int = 0
        self._step_counter: int = 0
        self._qotile_y: int = 10
        self._qotile_alive: bool = True
        self._destroyer: AtariEntity | None = None
        self._cannon_shot: AtariEntity | None = None
        self._qotile_dir: int = 1
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-yarsrevenge-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._step_counter = 0
        self._energy = 0
        self._qotile_alive = True
        self._destroyer = None
        self._cannon_shot = None
        self._qotile_dir = 1

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Shield wall
        self._shield_cells = set()
        for y in range(3, self._HEIGHT - 3):
            for dx in range(2):
                sx = self._SHIELD_X + dx
                self._shield_cells.add((sx, y))

        # Qotile position
        self._qotile_y = self._HEIGHT // 2

        # Player starts left side
        self._player_x = 3
        self._player_y = self._HEIGHT // 2

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 2:
            self._player_y += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._player_dir = (1, 0)

        # Eat shield cells (player touches them) -- only builds
        # energy; no reward. Reward is paid only on Qotile kills.
        pos = (self._player_x, self._player_y)
        if pos in self._shield_cells:
            self._shield_cells.discard(pos)
            self._energy = min(self._MAX_ENERGY, self._energy + 1)
            self._message = f"Shield eaten! Energy: {self._energy}"

        # Fire Zorlon cannon
        if action_name == "FIRE" and self._energy >= self._MAX_ENERGY and self._cannon_shot is None:
            self._cannon_shot = self._add_entity(
                "cannon", "=", self._player_x + 1,
                self._player_y, dx=1
            )
            self._energy = 0

        # Move cannon shot
        if self._cannon_shot is not None and self._cannon_shot.alive:
            self._cannon_shot.x += self._cannon_shot.dx
            cx, cy = self._cannon_shot.x, self._cannon_shot.y
            if cx >= self._WIDTH - 1:
                self._cannon_shot.alive = False
                self._cannon_shot = None
            elif (cx, cy) in self._shield_cells:
                # Cannon blocked by shield
                self._shield_cells.discard((cx, cy))
                self._cannon_shot.alive = False
                self._cannon_shot = None
                self._message = "Cannon blocked by shield!"
            elif (
                self._qotile_alive
                and cx == self._QOTILE_X
                and cy == self._qotile_y
            ):
                # Hit Qotile! One progress cycle complete.
                self._qotile_alive = False
                self._cannon_shot.alive = False
                self._cannon_shot = None
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Qotile destroyed!"
                if self._progress_count >= self._WIN_TARGET:
                    self._game_over = True
                    info["won"] = True
                    self._message = "All Qotile cycles cleared!"
                    return reward, self._game_over, info
                self._level += 1
                saved_progress = self._progress_count
                self._generate_level(self._level)
                self._progress_count = saved_progress
                return reward, self._game_over, info

        # Move Qotile
        if self._qotile_alive and self._step_counter % 3 == 0:
            self._qotile_y += self._qotile_dir
            if self._qotile_y <= 2 or self._qotile_y >= self._HEIGHT - 3:
                self._qotile_dir = -self._qotile_dir

        # Destroyer missile (homing)
        if (
            self._destroyer is None
            and self._step_counter % 15 == 0
            and self._qotile_alive
        ):
            self._destroyer = self._add_entity(
                "destroyer", "D", self._QOTILE_X - 1,
                self._qotile_y
            )

        if self._destroyer is not None and self._destroyer.alive:
            # Home toward player
            if self._step_counter % 2 == 0:
                if self._destroyer.x > self._player_x:
                    self._destroyer.x -= 1
                elif self._destroyer.x < self._player_x:
                    self._destroyer.x += 1
                if self._destroyer.y > self._player_y:
                    self._destroyer.y -= 1
                elif self._destroyer.y < self._player_y:
                    self._destroyer.y += 1

            # Hit player -- terminal failure
            if (
                self._destroyer.x == self._player_x
                and self._destroyer.y == self._player_y
            ):
                self._destroyer.alive = False
                self._destroyer = None
                self._on_life_lost()
                self._message = "Destroyed! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info

            # Out of bounds or enters neutral zone
            if self._destroyer is not None and self._destroyer.alive:
                dx, dy = self._destroyer.x, self._destroyer.y
                oob = dx <= 0 or dx >= self._WIDTH - 1
                oob = oob or dy <= 0 or dy >= self._HEIGHT - 1
                nz = self._NEUTRAL_LEFT <= dx <= self._NEUTRAL_RIGHT
                if oob or nz:
                    self._destroyer.alive = False
                    self._destroyer = None

        self._redraw()
        info["energy"] = self._energy
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if self._NEUTRAL_LEFT <= x <= self._NEUTRAL_RIGHT:
                    self._set_cell(x, y, ":")
                else:
                    self._set_cell(x, y, " ")

        # Shield
        for sx, sy in self._shield_cells:
            self._set_cell(sx, sy, "█")

        # Qotile
        if self._qotile_alive:
            self._set_cell(self._QOTILE_X, self._qotile_y, "Q")

        # Destroyer
        if self._destroyer is not None and self._destroyer.alive:
            self._set_cell(
                self._destroyer.x, self._destroyer.y, "D"
            )

        # Cannon shot
        if (
            self._cannon_shot is not None
            and self._cannon_shot.alive
        ):
            self._set_cell(
                self._cannon_shot.x, self._cannon_shot.y, "="
            )

    def _render_current_observation(self) -> Any:
        """Append yarsrevenge-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation as GO

        obs = super()._render_current_observation()
        extra = (
            f"Energy: {self._energy}/{self._MAX_ENERGY}"
        )
        new_hud = obs.hud + "\n" + extra
        return GO(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            ":": "neutral zone (safe from missile)",
            "█": "shield (eat for energy)",
            "Q": "Qotile (target)",
            "D": "destroyer missile (homing)",
            "=": "Zorlon cannon shot",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Eat shield cells (#) to build energy. "
            "When energy is full, FIRE the Zorlon cannon to "
            "hit the Qotile (Q) through the gap. "
            "Dodge the Destroyer missile. "
            "The neutral zone (:) is safe from missiles."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Yars' Revenge.\n\n"
            "TASK\n"
            "Eat the shield wall protecting the Qotile to build "
            "up energy, then fire the Zorlon cannon through the "
            "gap to destroy the Qotile. Dodge the homing Destroyer "
            "missile in the meantime.\n\n"
            "BOARD\n"
            "20x20 arena. Walls '-' and '|'. A shield wall '#' "
            "protects the Qotile 'Q' on one side of the arena. A "
            "neutral zone ':' runs vertically through the middle "
            "(Destroyer cannot enter). Destroyer missile 'D'. Your "
            "Zorlon cannon shot '='. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT move 1 cell in bounds. "
            "Walking onto a shield cell destroys it and gives +1 "
            "energy. FIRE with energy == 6 launches the cannon "
            "shot rightward from your row; the shot travels 1 "
            "cell per step and dies when it exits, hits shield, "
            "or hits Qotile. Qotile moves up/down every 3 steps. "
            "Destroyer spawns every 15 steps and homes onto you "
            "(moves every 2 steps); it dies if it enters the "
            "neutral zone.\n\n"
            "SCORING\n"
            "Pattern D: +1/3 reward per Qotile kill cycle "
            "(shield-eat to fill energy then hit the Qotile). "
            "Eating shield cells emits no reward (only energy). "
            "-1 reward on Destroyer contact. Cumulative reward "
            "bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: Destroyer contact ends with -1. "
            "Completing 3 Qotile kill cycles ends with cumulative "
            "+1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, and energy (out of 6).\n\n"
            + self.action_spec.render_for_prompt()
        )
