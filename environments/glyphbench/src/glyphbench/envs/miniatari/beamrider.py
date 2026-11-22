"""miniatari Beam Rider.

Identity: Ship rides 5 vertical beams shooting enemies that descend.
Win condition: clear 1 sector (10 enemies destroyed).
Reward: Pattern D, +1/10 per kill, -1 on enemy collision.
Loss: collision with descending enemy (terminal -1).

Tuning (Phase 3 hardening): win target bumped from 5 to 10. ~30% of
spawns are saucer enemies (S) that take 3 FIREs each — so a smart
agent must commit ~3-4 ticks of accurate fire on a saucer to clear it.
Random no longer accidentally clears the sector.

Gym ID: glyphbench/miniatari-beamrider-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniBeamRiderEnv(MiniatariBase):
    """Mini Beam Rider: 14x12 grid, 5 vertical beams.

    Beams are vertical guides at columns 2, 5, 7, 9, 12. The player ship
    snaps to one beam (LEFT/RIGHT cycles between beams). Enemies (E)
    spawn at row 0 on a random beam and descend at 1 cell every 2 ticks.
    FIRE shoots a torpedo straight up the player's beam, destroying the
    lowest enemy on that beam. Clear 10 enemies to win. Collision with
    an enemy (enemy reaches player row in player's beam) is -1 terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "snap to the next beam to the left",
            "snap to the next beam to the right",
            "fire a torpedo straight up your beam",
        ),
    )

    default_max_turns = 300

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 10
    _BEAMS = (2, 5, 7, 9, 12)
    _PLAYER_Y = 11
    _ENEMY_MOVE_EVERY = 2
    _SPAWN_PROB = 0.6
    _SAUCER_PROB = 0.30  # fraction of new spawns that are 3-HP saucers
    _FIRE_COOLDOWN = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._beam_idx: int = 0
        self._enemies: list[list[int]] = []  # [beam_idx, y]
        self._tick_count: int = 0
        self._progress: int = 0
        self._fire_cd: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-beamrider-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._fire_cd = 0
        self._enemies = []
        self._beam_idx = len(self._BEAMS) // 2
        self._player_x = self._BEAMS[self._beam_idx]
        self._player_y = self._PLAYER_Y
        # Pre-seed 2 enemies on distinct (beam, y) cells (no collisions).
        # Each entry is [beam_idx, y, hp]; hp=1 = standard, hp=3 = saucer.
        rng = self.rng
        used: set[tuple[int, int]] = set()
        for _ in range(2):
            for _attempt in range(40):
                bi = int(rng.integers(0, len(self._BEAMS)))
                y = int(rng.integers(0, 4))
                if (bi, y) in used:
                    continue
                used.add((bi, y))
                hp = 3 if rng.random() < self._SAUCER_PROB else 1
                self._enemies.append([bi, y, hp])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # 1. Beam change
        if action_name == "LEFT" and self._beam_idx > 0:
            self._beam_idx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._beam_idx < len(self._BEAMS) - 1:
            self._beam_idx += 1
            self._player_dir = (1, 0)
        self._player_x = self._BEAMS[self._beam_idx]

        # 2. Fire (instantaneous beam shot — damages lowest enemy in beam)
        if action_name == "FIRE" and self._fire_cd == 0:
            self._fire_cd = self._FIRE_COOLDOWN
            target: int | None = None
            target_y: int = -1
            for i, e in enumerate(self._enemies):
                eb, ey = e[0], e[1]
                if eb == self._beam_idx and ey > target_y and ey < self._PLAYER_Y:
                    target = i
                    target_y = ey
            if target is not None:
                self._enemies[target][2] -= 1
                if self._enemies[target][2] <= 0:
                    self._enemies.pop(target)
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._progress += 1
                    self._message = (
                        f"Enemy down! ({self._progress}/{self._WIN_TARGET})"
                    )
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
                else:
                    self._message = "Hit! (saucer takes more)"
        elif action_name == "FIRE":
            self._message = f"Torpedo recharging ({self._fire_cd} ticks)"

        # 3. Enemies descend
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            survivors = []
            for e in self._enemies:
                e[1] += 1
                if e[1] < self._PLAYER_Y:
                    survivors.append(e)
                elif e[1] == self._PLAYER_Y and e[0] == self._beam_idx:
                    # Collision
                    self._message = "Enemy crashed into your ship!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
                # else: enemy passed without hitting (different beam) -> remove silently
            self._enemies = survivors

        # 4. Spawn enemy
        rng = self.rng
        if rng.random() < self._SPAWN_PROB:
            # Pick a beam, avoid stacking at row 0
            occupied = {e[0] for e in self._enemies if e[1] == 0}
            free_beams = [i for i in range(len(self._BEAMS)) if i not in occupied]
            if free_beams:
                bi = free_beams[int(rng.integers(0, len(free_beams)))]
                hp = 3 if rng.random() < self._SAUCER_PROB else 1
                self._enemies.append([bi, 0, hp])

        info["progress"] = self._progress
        info["enemies_in_play"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Beams
        for y in range(self._HEIGHT):
            for bx in self._BEAMS:
                grid[y][bx] = "│"
        # Enemies — saucers (HP>1) shown as 'S', standard as 'E'.
        for e in self._enemies:
            eb, ey, hp = e[0], e[1], e[2]
            x = self._BEAMS[eb]
            if 0 <= x < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][x] = "S" if hp > 1 else "E"
        # Player ship
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "void",
            "│": "beam",
            "E": "enemy (1 HP)",
            "S": "saucer (3 HP, takes 3 hits)",
            "Y": "your ship",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Enemies: {len(self._enemies)}    "
            f"Cooldown: {self._fire_cd}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Beam Rider on a 14x12 grid with 5 vertical beams (│) "
            "at columns 2, 5, 7, 9, 12. Your ship (Y) snaps to one beam "
            "at row 11. LEFT/RIGHT cycles to adjacent beams. FIRE "
            "shoots a torpedo straight up your beam and damages the "
            "lowest enemy on that beam by 1 HP. FIRE has a 3-tick "
            "cooldown; firing while Cooldown is above 0 is a no-op. "
            "Standard enemies (E) "
            "have 1 HP — one FIRE destroys them. Saucers (S) have 3 HP "
            "— they need 3 FIREs each. Enemies spawn at the top on a "
            "random beam and descend 1 row every 2 ticks. Destroying "
            "10 enemies wins the sector. If an enemy reaches your row "
            "in your beam, you take a -1 terminal penalty. Reward: "
            "+1/10 per kill, -1 on collision."
        )
