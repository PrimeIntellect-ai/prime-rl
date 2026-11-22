"""Atari KungFuMaster environment.

Side-scrolling beat-em-up across 5 floors of a temple.

Gym ID: glyphbench/atari-kungfumaster-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 30
_H = 16
_FLOOR_Y = 12
_PLAYER_START_X = 2

class KungFuMasterEnv(AtariBase):
    """KungFuMaster: side-scrolling beat-em-up.

    Fight through 5 floors of enemies. Punch and kick
    grippers (G) and knife throwers (K). Reach the boss (B)
    at the right end of each floor.

    Grid: 30x16.
    Pattern D: +1/_WIN_TARGET per enemy or boss defeated
    (full-scope = 5 floors x 8 enemies = 40). -1.0 on KO.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT",
            "PUNCH", "KICK", "JUMP", "DUCK",
        ),
        descriptions=(
            "do nothing",
            "walk left",
            "walk right",
            "punch (short range)",
            "kick (medium range)",
            "jump up",
            "duck (dodge high attacks)",
        ),
    )

    # Pattern D full-scope target: 40 (5 floors x 8 enemies).
    _WIN_TARGET: int = 40
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._floor: int = 1
        self._ducking: bool = False
        self._jumping: bool = False
        self._jump_timer: int = 0
        self._spawn_timer: int = 0
        self._boss_spawned: bool = False
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-kungfumaster-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1013
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._ducking = False
        self._jumping = False
        self._jump_timer = 0
        self._spawn_timer = 0
        self._boss_spawned = False

        # Top/bottom borders
        for x in range(_W):
            self._set_cell(x, 0, "─")
            self._set_cell(x, _H - 1, "─")

        # Floor label area (top rows)
        floor_label = f" FLOOR {self._floor} "
        lx = (_W - len(floor_label)) // 2
        for i, ch in enumerate(floor_label):
            if 0 <= lx + i < _W:
                self._set_cell(lx + i, 1, ch)

        # Ground platform
        for x in range(_W):
            self._set_cell(x, _FLOOR_Y, "=")

        # Side walls
        self._set_cell(0, _FLOOR_Y - 1, "│")
        for y in range(2, _FLOOR_Y):
            self._set_cell(0, y, "│")
            self._set_cell(_W - 1, y, "│")

        # Decorative pillars
        for px in range(6, _W - 4, 8):
            for py in range(_FLOOR_Y - 3, _FLOOR_Y):
                self._set_cell(px, py, ":")

        # Spawn initial enemies
        n_enemies = 3 + self._floor
        for _i in range(n_enemies):
            ex = int(rng.integers(10, _W - 3))
            side = 1 if rng.random() < 0.5 else -1
            etype = "gripper" if rng.random() < 0.6 else "knife"
            ch = "G" if etype == "gripper" else "K"
            e = self._add_entity(
                etype, ch, ex, _FLOOR_Y - 1, dx=-side
            )
            e.data["timer"] = int(rng.integers(3, 8))

        # Player start
        self._player_x = _PLAYER_START_X
        self._player_y = _FLOOR_Y - 1

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._ducking = False
        attack = ""

        if action_name == "LEFT":
            if self._player_x > 1:
                self._player_x -= 1
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            if self._player_x < _W - 2:
                self._player_x += 1
                self._player_dir = (1, 0)
        elif action_name == "PUNCH":
            attack = "punch"
        elif action_name == "KICK":
            attack = "kick"
        elif action_name == "JUMP":
            if not self._jumping:
                self._jumping = True
                self._jump_timer = 4
                self._player_y = _FLOOR_Y - 3
        elif action_name == "DUCK":
            self._ducking = True

        # Jump physics
        if self._jumping:
            self._jump_timer -= 1
            if self._jump_timer <= 0:
                self._jumping = False
                self._player_y = _FLOOR_Y - 1

        # Attack: hit nearby enemies
        if attack:
            reach = 1 if attack == "punch" else 2
            for e in self._entities:
                if e.etype in ("gripper", "knife", "boss"):
                    if not e.alive:
                        continue
                    if (
                        abs(e.x - self._player_x) <= reach
                        and abs(e.y - self._player_y) <= 1
                    ):
                        if e.etype == "boss":
                            e.data["hp"] = (
                                e.data.get("hp", 5) - 1
                            )
                            if e.data["hp"] <= 0:
                                e.alive = False
                                if self._progress_count < self._WIN_TARGET:
                                    reward += 1.0 / self._WIN_TARGET
                                    self._progress_count += 1
                                self._message = (
                                    "Boss defeated!"
                                )
                        else:
                            e.alive = False
                            if self._progress_count < self._WIN_TARGET:
                                reward += 1.0 / self._WIN_TARGET
                                self._progress_count += 1
                            self._message = (
                                f"{e.etype} defeated!"
                            )

        # Enemy AI
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "gripper":
                e.data["timer"] = (
                    e.data.get("timer", 3) - 1
                )
                if e.data["timer"] <= 0:
                    e.data["timer"] = int(
                        self.rng.integers(2, 5)
                    )
                    # Move toward player
                    if e.x > self._player_x:
                        e.x -= 1
                    elif e.x < self._player_x:
                        e.x += 1
            elif e.etype == "knife":
                e.data["timer"] = (
                    e.data.get("timer", 4) - 1
                )
                if e.data["timer"] <= 0:
                    e.data["timer"] = int(
                        self.rng.integers(3, 7)
                    )
                    # Throw knife projectile
                    kdir = (
                        -1 if e.x > self._player_x else 1
                    )
                    self._add_entity(
                        "projectile", "~", e.x + kdir,
                        e.y, dx=kdir,
                    )
                    # Also approach slowly
                    if self.rng.random() < 0.3:
                        if e.x > self._player_x:
                            e.x -= 1
                        else:
                            e.x += 1
            elif e.etype == "boss":
                e.data["timer"] = (
                    e.data.get("timer", 5) - 1
                )
                if e.data["timer"] <= 0:
                    e.data["timer"] = int(
                        self.rng.integers(3, 6)
                    )
                    if e.x > self._player_x:
                        e.x -= 1
                    elif e.x < self._player_x:
                        e.x += 1

        # Move projectiles
        for e in self._entities:
            if e.etype == "projectile" and e.alive:
                e.x += e.dx
                if e.x <= 0 or e.x >= _W - 1:
                    e.alive = False

        # Collision: projectiles hit player (Pattern D death)
        for e in self._entities:
            if e.etype != "projectile" or not e.alive:
                continue
            if (
                e.x == self._player_x
                and abs(e.y - self._player_y) <= 1
            ):
                if self._ducking:
                    pass  # dodged by ducking
                else:
                    e.alive = False
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._message = "Hit by projectile!"
                    return reward, self._game_over, info

        # Collision: enemies touch player (Pattern D death)
        for e in self._entities:
            if e.etype not in ("gripper", "boss"):
                continue
            if not e.alive:
                continue
            if (
                e.x == self._player_x
                and abs(e.y - self._player_y) <= 1
            ):
                if self._jumping:
                    pass  # jumped over
                else:
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._message = f"Grabbed by {e.etype}!"
                    return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Spawn more enemies periodically
        self._spawn_timer += 1
        if self._spawn_timer >= max(15 - self._floor * 2, 5):
            self._spawn_timer = 0
            enemies_alive = sum(
                1 for e in self._entities
                if e.etype in ("gripper", "knife")
            )
            if enemies_alive < 3 + self._floor:
                side = (
                    _W - 2 if self.rng.random() < 0.5 else 1
                )
                etype = (
                    "gripper"
                    if self.rng.random() < 0.6
                    else "knife"
                )
                ch = "G" if etype == "gripper" else "K"
                self._add_entity(
                    etype, ch, side, _FLOOR_Y - 1
                )

        # Boss spawn when player reaches right side
        if (
            not self._boss_spawned
            and self._player_x >= _W - 6
        ):
            self._boss_spawned = True
            b = self._add_entity(
                "boss", "B", _W - 3, _FLOOR_Y - 1
            )
            b.data["hp"] = 3 + self._floor
            self._message = "Boss appeared!"

        # Floor cleared: all enemies dead + boss dead
        bosses = [
            e for e in self._entities
            if e.etype == "boss" and e.alive
        ]
        if self._boss_spawned and len(bosses) == 0:
            self._floor += 1
            if self._floor > 5:
                self._floor = 1
                self._level += 1
            self._message = f"Floor cleared! Floor {self._floor}"
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All floors cleared!"
            else:
                self._generate_level(self._level * 4001 + self._floor)

        # Win check (in case target hit between floor clears)
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All floors cleared!"

        info["floor"] = self._floor
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "│": "wall",
            "=": "floor",
            ":": "pillar",
            "G": "gripper enemy",
            "K": "knife thrower",
            "B": "boss",
            "~": "thrown knife",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        boss_hp = "none"
        for e in self._entities:
            if e.etype == "boss" and e.alive:
                boss_hp = str(e.data.get("hp", 0))
                break
        extra = (
            f"Floor: {self._floor}/5  Boss: {boss_hp}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Fight through 5 floors. PUNCH and KICK enemies. "
            "JUMP to dodge, DUCK to avoid knives. "
            "Defeat the boss (B) at each floor's end."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Kung Fu Master.\n\n"
            "TASK\n"
            "Fight side-scroller style through 5 floors of a temple. "
            "Each floor ends with a boss; beating the boss advances "
            "to the next floor, clearing all 5 advances the level.\n\n"
            "BOARD\n"
            "30 columns by 16 rows. Walls border the room; a floor "
            "'=' runs across the level with regularly-spaced pillars "
            "':'. Grippers 'G' grab you, knife throwers 'K' fling "
            "knives 'tilde'. Boss 'B' spawns when you reach the right "
            "edge. You are an arrow glyph standing on the floor.\n\n"
            "MECHANICS\n"
            "LEFT/RIGHT walk 1 cell. PUNCH hits enemies within "
            "distance 1 at your row. KICK hits within distance 2 at "
            "your row. JUMP leaps up 2 rows for 4 steps (lets you "
            "pass over grippers). DUCK dodges knives for that step. "
            "Grippers chase you every 2-5 steps on an axis. Knife "
            "throwers throw a 'tilde' projectile toward you every "
            "3-7 steps that travels 1 cell per step horizontally.\n\n"
            "SCORING\n"
            "+1/40 reward per Gripper, Knife thrower, or Boss "
            "defeated (Pattern D full-scope = 5 floors x 8 enemies "
            "= 40). -1.0 on KO (enemy or knife contact while not "
            "ducking/jumping).\n\n"
            "TERMINATION\n"
            "Contact with an enemy or a knife ends the episode "
            "with -1.0. Episode ends after 40 enemies defeated "
            "(cumulative reward plateaus at +1.0) or after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, current floor (1-5), and "
            "boss HP (or 'none').\n\n"
            + self.action_spec.render_for_prompt()
        )
