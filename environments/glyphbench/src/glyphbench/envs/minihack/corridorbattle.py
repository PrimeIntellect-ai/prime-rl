"""MiniHack CorridorBattle environments. Fight through monsters in a corridor."""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, ORC, RAT
from glyphbench.envs.minihack.items import POTION_HEALING, SWORD


class _CorridorBattleBase(MiniHackBase):
    _is_dark: bool = False
    _num_monsters: int = 4

    def _generate_level(self, seed: int) -> None:
        w, h = 15, 7
        self._init_grid(w, h)
        self._dark = self._is_dark

        # Fill everything with walls, then carve a corridor and one potion
        # alcove. Blind east-walking is intentionally unsafe; a skilled policy
        # should pick up and wield the sword before engaging.
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                self._place_wall(x, y)

        corridor_y = h // 2
        for x in range(1, w - 1):
            self._grid[corridor_y][x] = "·"
        self._grid[corridor_y - 1][8] = "·"

        self._place_player(1, corridor_y)
        self._place_stairs(w - 2, corridor_y)
        self._place_item(1, corridor_y, SWORD)
        self._place_item(8, corridor_y - 1, POTION_HEALING)

        guard_plan = [(RAT, 4), (KOBOLD, 7), (ORC, 10), (ORC, 12)]
        for ctype, mx in guard_plan[: self._num_monsters]:
            self._spawn_creature(ctype, mx, corridor_y)

    def _move_monsters(self) -> None:
        """Corridor guards hold position; retaliation is handled on attacks."""

    def _on_quaff_potion(self, potion) -> None:  # noqa: ANN001
        if potion.name == POTION_HEALING.name:
            self._player_hp = self._player_max_hp
            self._message += " Your wounds close."

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            monster = self._creature_at(nx, ny)
            if monster is not None:
                if self._wielding and self._wielding.name == SWORD.name:
                    dmg = 4
                else:
                    dmg = int(self.rng.integers(1, 3))
                monster.hp -= dmg
                self._message = (
                    f"You hit the {monster.ctype.name}! "
                    f"(-{dmg} HP, "
                    f"{monster.hp}/{monster.ctype.max_hp} remaining)"
                )
                if monster.hp <= 0:
                    self._message += f" The {monster.ctype.name} dies."
                    self._creatures = [c for c in self._creatures if c.hp > 0]
                    return self._render_current_observation(), 0.0, False, False, {
                        "player_pos": self._player_pos,
                        "hp": self._player_hp,
                    }

                self._player_hp -= max(1, monster.ctype.damage)
                self._message += f" The {monster.ctype.name} hits you back!"
                if self._player_hp <= 0:
                    self._message += " You die."
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "combat"
                    }
                return self._render_current_observation(), 0.0, False, False, {
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
        return super()._step(action)

    def _task_description(self) -> str:
        parts = [
            f"Fight through {self._num_monsters} monsters in a corridor "
            f"to reach the stairs (⇣). A long sword lies at your feet: "
            f"PICKUP and WIELD it before fighting. A healing potion is in a "
            f"side alcove before the orc guards."
        ]
        if self._is_dark:
            parts.append("The corridor is dark -- you can only see adjacent tiles.")
        parts.append(
            "Move into a monster to attack it. Reward: +1 on reaching stairs, -1 on death."
        )
        return " ".join(parts)


class MiniHackCorridorBattleEnv(_CorridorBattleBase):
    def env_id(self) -> str:
        return "glyphbench/minihack-corridorbattle-v0"


class MiniHackCorridorBattleDarkEnv(_CorridorBattleBase):
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-corridorbattle-dark-v0"
