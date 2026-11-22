"""MiniHack Quaff skill tasks.

Phase 3:
  * quaff: randomise potion + monster + player + stairs positions.
  * quaff-distract: C6 catalytic — drinking only heals AT the sink tile;
    quaffing elsewhere damages the player.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD
from glyphbench.envs.minihack.items import POTION_HEALING, Item


class _QuaffBase(MiniHackBase):
    _distract: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._sink_pos: tuple[int, int] | None = None
        self._quaffed: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._player_hp = 3
        self._quaffed = False
        # Build candidate cells and place player, stairs, potion, monster.
        interior = [(x, y) for y in range(1, 6) for x in range(1, 6)]
        idx = int(self.rng.integers(0, len(interior)))
        px, py = interior[idx]
        self._place_player(px, py)
        cands = [c for c in interior if c != (px, py)]
        sx, sy = cands[int(self.rng.integers(0, len(cands)))]
        self._place_stairs(sx, sy)
        cands = [c for c in cands if c != (sx, sy)]
        # Potion
        potion_pos = cands[int(self.rng.integers(0, len(cands)))]
        self._place_item(*potion_pos, POTION_HEALING)
        cands = [c for c in cands if c != potion_pos]
        # Kobold guarding (random in cands)
        kobold_pos = cands[int(self.rng.integers(0, len(cands)))]
        self._spawn_creature(KOBOLD, *kobold_pos)
        cands = [c for c in cands if c != kobold_pos]

        if self._distract:
            # Place a sink — quaffing elsewhere = damage, on sink = heal
            sink_pos = cands[int(self.rng.integers(0, len(cands)))]
            sx2, sy2 = sink_pos
            self._grid[sy2][sx2] = "{"
            self._sink_pos = sink_pos

    def _on_quaff_potion(self, potion: Item) -> None:
        if not self._distract:
            if potion.name == POTION_HEALING.name:
                self._player_hp = self._player_max_hp
                self._message += " You feel much better!"
                self._quaffed = True
            return
        # Distract: catalytic — must be on sink
        if self._player_pos == self._sink_pos:
            self._player_hp = self._player_max_hp
            self._message += " The sink water purifies the potion. You feel much better!"
            self._quaffed = True
        else:
            # Drinking elsewhere damages instead
            self._player_hp = max(0, self._player_hp - 2)
            self._message += " You drink the bitter potion. (-2 HP)"

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated and info.get("goal_reached") and not self._quaffed:
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "You are too weak to descend safely. Quaff first."
            obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        if self._distract:
            return (
                "You are near death. A monster guards the stairs. A potion of "
                "healing (!) and a sink ({) are scattered. PICKUP the potion "
                "and walk to the SINK before QUAFFing — drinking elsewhere is "
                "harmful (-2 HP). Then fight past the monster and reach the "
                "stairs (⇣). Reward: +1 stairs, -1 death."
            )
        return (
            "You are near death with a monster guarding the stairs. "
            "Pick up the potion of healing (!) and QUAFF it to restore HP, "
            "then fight the monster and reach the stairs (⇣). "
            "Reward: +1 stairs, -1 death."
        )


class MiniHackQuaffEnv(_QuaffBase):
    """MiniHack Quaff: drink a potion of healing to survive combat."""

    def env_id(self) -> str:
        return "glyphbench/minihack-quaff-v0"


class MiniHackQuaffDistractEnv(_QuaffBase):
    """MiniHack Quaff (Distract): C6 catalytic — must drink at sink."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-quaff-distract-v0"
