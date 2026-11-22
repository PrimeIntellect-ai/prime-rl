"""MiniHack Wand of Death skill tasks.

Phase 3 redesign — each variant has a distinct twist:
  * easy: G6 negotiator — single TROLL (huge HP / dmg) so the wand is
    mandatory; melee = certain death.
  * medium: E3 mana ration — wand has 2 charges total, 3 enemies; agent
    must aim shots.
  * hard: real harmful effects on fire/cold wands + obscure wands as
    "unidentified wand X" (X=1,2,3) until ZAP'd once. ZAPping reveals
    the identity. Agent must learn through trial.
  * pro: dark + 4 monsters + wand-id puzzle (medium + hard combined).

For the hard / pro variants the three wands render with DISTINCT
single-codepoint glyphs (/ ↯ ⌇) so the agent can track identifications
across turns. The legend hides each wand's true name behind the alias
'unidentified wand N' until that specific wand is ZAPped — only then is
its identity revealed.
"""

from __future__ import annotations

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, OGRE, ORC, TROLL, ZOMBIE
from glyphbench.envs.minihack.items import WAND_COLD, WAND_DEATH, WAND_FIRE, Item

# Per-episode Item instances are created in _generate_level so unidentified
# display names do not leak across resets.


class _WoDBase(MiniHackBase):
    _num_monsters: int = 1
    _has_distractors: bool = False
    _is_dark: bool = False
    _wand_charges: int = 1  # how many ZAPs the WoD has total
    _troll_only: bool = False  # if True, replace monsters with one troll
    _wand_id_puzzle: bool = False  # obscure wand identities until zapped

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._wod_charges_remaining: int = self._wand_charges
        self._wand_used: bool = False
        # Map wand-name → "unidentified wand 1/2/3"
        self._wand_aliases: dict[str, str] = {}
        self._revealed_wands: set[str] = set()

    def _generate_level(self, seed: int) -> None:
        self._init_grid(9, 9)
        self._dark = self._is_dark
        self._wod_charges_remaining = self._wand_charges
        self._wand_used = False
        self._wand_aliases = {}
        self._revealed_wands = set()

        # Wand-id puzzle: rename all wands as "unidentified wand 1/2/3"
        if self._wand_id_puzzle:
            ids = list(self.rng.permutation(3))
            wand_names = [WAND_DEATH.name, WAND_FIRE.name, WAND_COLD.name]
            for i, w_name in enumerate(wand_names):
                self._wand_aliases[w_name] = f"unidentified wand {ids[i] + 1}"

        def local_wand(item: Item, glyph: str) -> Item:
            return Item(item.name, glyph, "wand", self._wand_aliases[item.name])

        # Choose the per-floor placement item: when the wand-id puzzle is
        # active, use distinct-glyph local items so the agent can track
        # identifications across turns; otherwise keep the canonical '/'.
        wand_death_local = (
            local_wand(WAND_DEATH, "/") if self._wand_id_puzzle else WAND_DEATH
        )
        wand_fire_local = (
            local_wand(WAND_FIRE, "↯") if self._wand_id_puzzle else WAND_FIRE
        )
        wand_cold_local = (
            local_wand(WAND_COLD, "⌇") if self._wand_id_puzzle else WAND_COLD
        )

        # Player on left
        py = int(self.rng.integers(1, 8))
        self._place_player(1, py)
        # Stairs on right
        sy = int(self.rng.integers(1, 8))
        self._place_stairs(7, sy)

        # Wand of Death position
        wx = int(self.rng.integers(2, 5))
        wy = int(self.rng.integers(1, 8))
        self._place_item(wx, wy, wand_death_local)
        wand_positions = {(wx, wy)}

        # Distractor / decoy wands
        if self._has_distractors:
            for wand in (wand_fire_local, wand_cold_local):
                attempts = 0
                while attempts < 30:
                    dx = int(self.rng.integers(2, 5))
                    dy = int(self.rng.integers(1, 8))
                    if (dx, dy) not in wand_positions:
                        self._place_item(dx, dy, wand)
                        wand_positions.add((dx, dy))
                        break
                    attempts += 1

        # Monsters
        if self._troll_only:
            mtypes = [TROLL]
            self._num_monsters_actual = 1
        else:
            monster_types = [KOBOLD, ORC, ZOMBIE, OGRE]
            mtypes = [
                monster_types[i % len(monster_types)]
                for i in range(self._num_monsters)
            ]
            self._num_monsters_actual = self._num_monsters
        for ctype in mtypes:
            attempts = 0
            while attempts < 50:
                mx = int(self.rng.integers(5, 7))
                my = int(self.rng.integers(1, 8))
                if self._creature_at(mx, my) is None and (mx, my) not in wand_positions:
                    self._spawn_creature(ctype, mx, my)
                    break
                attempts += 1

    def _wand_display_name(self, wand: Item) -> str:
        if self._wand_id_puzzle and wand.name not in self._revealed_wands:
            return self._wand_aliases.get(wand.name, wand.name)
        return wand.name

    def _refresh_wand_display_names(self) -> None:
        if not self._wand_id_puzzle:
            return
        items: list[Item] = list(self._inventory)
        for stack in self._floor_items.values():
            items.extend(stack)
        for item in items:
            if item.item_type == "wand" and item.name in self._wand_aliases:
                item.display_name = (
                    None
                    if item.name in self._revealed_wands
                    else self._wand_aliases[item.name]
                )

    def _alias_unrevealed_text(self, text: str) -> str:
        if not self._wand_id_puzzle:
            return text
        for w_name, alias in self._wand_aliases.items():
            if w_name not in self._revealed_wands:
                text = text.replace(w_name, alias)
        return text

    def _on_zap_wand(self, wand: Item) -> None:
        # Reveal the wand name on first zap (wand-id puzzle)
        if wand.name in self._wand_aliases and wand.name not in self._revealed_wands:
            alias = self._wand_aliases.get(wand.name, wand.name)
            self._revealed_wands.add(wand.name)
            self._refresh_wand_display_names()
            self._message += f" The {alias} reveals itself as {wand.name}!"

        if wand.name == WAND_DEATH.name:
            if self._wod_charges_remaining <= 0:
                self._message += " The wand is depleted — no effect."
                return
            self._wod_charges_remaining -= 1
            # Kill nearest visible monster
            px, py = self._player_pos
            nearest = None
            nearest_dist = float("inf")
            for c in self._creatures:
                if c.hp <= 0:
                    continue
                dist = abs(c.x - px) + abs(c.y - py)
                if dist < nearest_dist:
                    nearest = c
                    nearest_dist = dist
            if nearest is not None:
                nearest.hp = 0
                self._message += f" The {nearest.ctype.name} is killed!"
                self._creatures = [c for c in self._creatures if c.hp > 0]
        elif wand.name == WAND_FIRE.name:
            # Real harmful effect: burn self
            self._player_hp = max(0, self._player_hp - 2)
            self._message += " The wand of fire backfires! (-2 HP)"
        elif wand.name == WAND_COLD.name:
            self._player_hp = max(0, self._player_hp - 2)
            self._message += " The wand of cold chills you! (-2 HP)"

    def _step(self, action: int):
        name = self.action_spec.names[action]
        if name == "ZAP":
            wand = next(
                (i for i in self._inventory if i.item_type == "wand"), None
            )
            if wand is not None:
                self._message = f"You zap the {self._wand_display_name(wand)}!"
                self._wand_used = True
                self._on_zap_wand(wand)
                if self._player_hp <= 0:
                    self._message = (self._message + " You die.").strip()
                    return (
                        self._render_current_observation(),
                        -1.0,
                        True,
                        False,
                        {"cause_of_death": "hazard"},
                    )
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
        if (
            terminated
            and info.get("goal_reached")
            and self._troll_only
            and not self._wand_used
        ):
            terminated = False
            reward = 0.0
            info.pop("goal_reached", None)
            self._message = "You should zap the wand of death before descending."
            obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        self._refresh_wand_display_names()
        obs = super()._render_current_observation()
        extras: list[str] = []
        if self._wand_charges > 1:
            extras.append(
                f"WoD charges remaining: {self._wod_charges_remaining}/"
                f"{self._wand_charges}"
            )
        if self._troll_only:
            extras.append(f"Wand used: {'yes' if self._wand_used else 'no'}")
        if self._wand_id_puzzle:
            # Show alias mapping for revealed wands
            revealed = (
                "; ".join(
                    f"{self._wand_aliases[n]} = {n}"
                    for n in self._revealed_wands
                )
                if self._revealed_wands
                else "(none yet)"
            )
            extras.append(f"Wand IDs known: {revealed}")
        new_hud = obs.hud + "    " + "    ".join(extras) if extras else obs.hud
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=self._alias_unrevealed_text(new_hud),
            message=self._alias_unrevealed_text(obs.message),
        )

    def _task_description(self) -> str:
        parts = []
        if self._troll_only:
            parts.append(
                "A TROLL (T, 20 HP, 8 dmg) blocks the path. Melee will kill "
                "you in two hits — the only survivable strategy is to find "
                "the wand of death (/) and ZAP it before descending."
            )
        else:
            if self._wand_id_puzzle:
                parts.append(
                    f"A room with {self._num_monsters} hostile monster(s). "
                    "Three unidentified wands with distinct glyphs are on the "
                    "floor. Pick up a wand with PICKUP and use ZAP to identify "
                    "it and apply its effect."
                )
            else:
                parts.append(
                    f"A room with {self._num_monsters} hostile monster(s). "
                    "Find the wand of death (/) on the floor, pick it up with "
                    "PICKUP, then use ZAP to kill monsters."
                )
        if self._wand_charges > 1:
            parts.append(
                f"The wand has {self._wand_charges} charges total — choose "
                "your shots."
            )
        if self._wand_id_puzzle:
            parts.append(
                "All wands are 'unidentified wand X' until ZAP'd once. "
                "Fire/cold wands HURT you (-2 HP each); only the WoD kills "
                "monsters. Read the wand alias HUD to track identifications."
            )
        if self._is_dark:
            parts.append(
                "The room is dark — you can only see adjacent tiles."
            )
        parts.append("Reach the stairs (⇣). Reward: +1 stairs, -1 death.")
        return " ".join(parts)


class MiniHackWoDEasyEnv(_WoDBase):
    """G6 negotiator — single troll, wand mandatory."""

    _num_monsters = 1
    _troll_only = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-easy-v0"


class MiniHackWoDMediumEnv(_WoDBase):
    """E3 mana ration — 2 charges, 3 enemies."""

    _num_monsters = 3
    _wand_charges = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-medium-v0"


class MiniHackWoDHardEnv(_WoDBase):
    """Real harmful fire/cold + wand-id puzzle."""

    _num_monsters = 3
    _has_distractors = True
    _wand_id_puzzle = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-hard-v0"


class MiniHackWoDProEnv(_WoDBase):
    """Dark + 4 monsters + wand-id puzzle (medium + hard combined)."""

    _num_monsters = 4
    _has_distractors = True
    _wand_id_puzzle = True
    _wand_charges = 2
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-pro-v0"
