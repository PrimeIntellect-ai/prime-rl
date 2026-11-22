"""MiniHack Wield skill tasks.

Phase 3:
  * wield: randomise sword + monster + player + stairs positions.
  * wield-distract: H6 decoy legend — three weapons on the floor:
    one real sword `)`, one rusty sword `≀` (zero damage), one club
    `▏` (modest damage). Each weapon has a distinct single-codepoint
    glyph and a distinct legend entry. The HUD describes the rusty
    sword's drawback but does NOT name the real weapon — the agent
    must read the legend to find the non-rusty long sword.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD
from glyphbench.envs.minihack.items import SWORD, Item

# Decoy weapons used by the distract variant. Distinct single-codepoint
# glyphs ensure the legend can list each by its proper name.
RUSTY_SWORD = Item("rusty long sword", "≀", "weapon")
CLUB = Item("wooden club", "▏", "weapon")


class _WieldBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        # Player at random
        interior = [(x, y) for y in range(1, 6) for x in range(1, 6)]
        idx = int(self.rng.integers(0, len(interior)))
        px, py = interior[idx]
        self._place_player(px, py)
        # Stairs
        cands = [c for c in interior if c != (px, py)]
        sx, sy = cands[int(self.rng.integers(0, len(cands)))]
        self._place_stairs(sx, sy)
        cands = [c for c in cands if c != (sx, sy)]

        if not self._distract:
            # Sword
            sword_pos = cands[int(self.rng.integers(0, len(cands)))]
            self._place_item(*sword_pos, SWORD)
            cands = [c for c in cands if c != sword_pos]
            # Two kobolds
            for _ in range(2):
                if not cands:
                    break
                pos = cands[int(self.rng.integers(0, len(cands)))]
                self._spawn_creature(KOBOLD, *pos)
                cands = [c for c in cands if c != pos]
        else:
            # Place 3 weapons: real sword, rusty sword, club
            weapons = [SWORD, RUSTY_SWORD, CLUB]
            shuffled = [cands[i] for i in self.rng.permutation(len(cands))[:3]]
            for w, pos in zip(weapons, shuffled, strict=True):
                self._place_item(*pos, w)
            cands = [c for c in cands if c not in shuffled]
            # Two kobolds
            for _ in range(2):
                if not cands:
                    break
                pos = cands[int(self.rng.integers(0, len(cands)))]
                self._spawn_creature(KOBOLD, *pos)
                cands = [c for c in cands if c != pos]

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            monster = self._creature_at(nx, ny)
            if monster is not None and self._wielding is None:
                self._message = "You need to wield a weapon before attacking."
                self._move_monsters()
                if self._player_hp <= 0:
                    self._message = (self._message + " You die.").strip()
                    return self._render_current_observation(), -1.0, True, False, {
                        "cause_of_death": "monster"
                    }
                return self._render_current_observation(), 0.0, False, False, {
                    "player_pos": self._player_pos,
                    "hp": self._player_hp,
                }
            if monster is not None and self._wielding is not None:
                weapon_name = self._wielding.name
                # Damage depends on weapon
                if weapon_name == SWORD.name:
                    dmg = max(1, int(self.rng.integers(6, 11)))
                elif weapon_name == CLUB.name:
                    dmg = max(1, int(self.rng.integers(2, 5)))
                else:  # rusty sword
                    dmg = 0
                if dmg > 0:
                    monster.hp -= dmg
                    self._message = (
                        f"You slash the {monster.ctype.name} with your "
                        f"{weapon_name}! (-{dmg} HP)"
                    )
                else:
                    self._message = (
                        f"You swing the {weapon_name} but it does no damage!"
                    )
                if monster.hp <= 0:
                    self._message += f" The {monster.ctype.name} dies."
                    self._creatures = [c for c in self._creatures if c.hp > 0]

                self._move_monsters()
                if self._player_hp <= 0:
                    self._message = (self._message + " You die.").strip()
                    return (
                        self._render_current_observation(),
                        -1.0,
                        True,
                        False,
                        {"cause_of_death": "monster"},
                    )

                if self._goal_pos and self._player_pos == self._goal_pos:
                    if self._wielding is None:
                        self._message = (
                            "You should wield a weapon before descending."
                        )
                        return (
                            self._render_current_observation(),
                            0.0,
                            False,
                            False,
                            {"player_pos": self._player_pos, "hp": self._player_hp},
                        )
                    self._message = "You reach the stairs. You descend."
                    return (
                        self._render_current_observation(),
                        1.0,
                        True,
                        False,
                        {
                            "goal_reached": True,
                            "player_pos": self._player_pos,
                            "hp": self._player_hp,
                        },
                    )

                return (
                    self._render_current_observation(),
                    0.0,
                    False,
                    False,
                    {"player_pos": self._player_pos, "hp": self._player_hp},
                )

        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated and info.get("goal_reached") and self._wielding is None:
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "You should wield a weapon before descending."
            obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._distract:
            # H6 hint — describe ONLY the decoy. Don't name the real weapon.
            extra = (
                "    Hint: the 'rusty long sword' (≀) is corroded — it does "
                "no damage. Read the legend to identify which weapon is which."
            )
            new_hud = obs.hud + extra
            return GridObservation(
                grid=obs.grid,
                legend=obs.legend,
                hud=new_hud,
                message=obs.message,
            )
        return obs

    def _task_description(self) -> str:
        if self._distract:
            return (
                "Monsters guard the stairs. Three weapons lie on the floor; "
                "the legend names each by its glyph. ONE of them is corroded "
                "and useless — the 'rusty long sword' does NO damage. Read "
                "the legend, AVOID the rusty weapon, PICKUP a usable one and "
                "WIELD it before fighting. Reward: +1 stairs, -1 death."
            )
        return (
            "Monsters guard the stairs. Pick up the long sword glyph ) and WIELD it "
            "for much higher damage, then fight through and reach the stairs (⇣). "
            "Reward: +1 stairs, -1 death."
        )


class MiniHackWieldEnv(_WieldBase):
    """MiniHack Wield: wield a weapon for higher combat damage."""

    def env_id(self) -> str:
        return "glyphbench/minihack-wield-v0"


class MiniHackWieldDistractEnv(_WieldBase):
    """MiniHack Wield (Distract): H6 decoy legend (rusty sword, club)."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wield-distract-v0"
