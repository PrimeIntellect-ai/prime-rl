"""Extended Craftax sub-task environments (35 focused tasks).

Each environment subclasses CraftaxScenarioEnv or CraftaxFullEnv and
overrides ``_reset`` to set up a curated initial state, and ``_step``
to check sub-task-specific termination and reward.
All registered tasks use the single Craftax mechanics contract and universal
Craftax action space; focused starts narrow only the scenario state and reward.

Gym IDs: glyphbench/craftax-<taskname>-v0
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    TILE_CHEST,
    TILE_COAL,
    TILE_DIAMOND,
    TILE_DUNGEON_FLOOR,
    TILE_ENCHANT_FIRE,
    TILE_ENCHANT_ICE,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_IRON,
    TILE_PLACED_STONE,
    TILE_RIPE_PLANT,
    TILE_SAND,
    TILE_STAIRS_DOWN,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_WATER,
)
from glyphbench.envs.craftax.scenario import (
    _DAY_LENGTH,
    _MAX_ENERGY,
    _MAX_FOOD,
    _MAX_WATER,
    _MOB_STATS,
    CraftaxScenarioEnv,
    Mob,
)
from glyphbench.envs.craftaxfull.full import (
    _DUNGEON_SIZE,
    _MAX_MANA,
    _NIGHT_LENGTH,
    _SURFACE_SIZE,
    DUNGEON_WALKABLE,
    SURFACE_WALKABLE,
    CraftaxFullEnv,
)
from glyphbench.envs.craftaxfull.full import (
    _MOB_STATS as FULL_MOB_STATS,
)
from glyphbench.envs.craftaxfull.full import Mob as FullMob

_FLOOR_NAV_BASE_SECTIONS = (
    "overview",
    "legend:player", "legend:terrain", "legend:mobs:dungeon",
    "legend:items", "legend:projectiles", "legend:hud",
    "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
    "combat:melee", "combat:ranged_player", "combat:ranged_mob",
    "combat:armor", "combat:projectiles",
)

_MAGIC_COMBAT_SECTIONS = (
    "overview",
    "legend:player", "legend:terrain", "legend:mobs:dungeon",
    "legend:items", "legend:projectiles", "legend:hud",
    "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
    "survival:mana",
    "combat:melee", "combat:ranged_player", "combat:ranged_mob",
    "combat:armor", "combat:projectiles", "combat:elemental",
    "magic:spells", "magic:enchants", "items:potions",
)

_SURFACE_COMBAT_SECTIONS = (
    "overview",
    "legend:player", "legend:terrain", "legend:mobs:overworld",
    "legend:projectiles", "legend:hud",
    "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
    "combat:melee", "combat:ranged_player", "combat:ranged_mob",
    "combat:armor", "combat:projectiles",
    "items:bow", "items:potions", "floors:0",
)

# Floor-exit kill gate. Re-tuned (was 8 in commit 58f0fde, which made every
# _FloorExitGateMixin floor unsolvable at the 9-HP nav budget — a skilled
# kite-and-attack policy died before clearing the gate on every seed). A
# 3-kill gate restores a survivable-but-mandatory combat objective: a trivial
# policy (random / always-DO / blind DESCEND) still dies far short of it, while
# a skilled agent that isolates and kills 3 mobs then runs to the stairs can
# win. Paired with the raised floor-nav HP budget below.
_FLOOR_GATE_KILLS = 3

# Floor-navigation HP budget. The shared 9-HP cap (BASE_MAX_HP) gives a literal
# 2-hit death against the deeper floors' 6-9 dmg/hit mobs, so even a perfect
# kite-and-attack policy cannot survive the mandatory kill gate. These per-floor
# values are set in each env's _reset (after super()._reset, so they are durable
# — _recompute_max_stats only runs on reset/LEVEL_UP and the nav envs grant 0 XP).
# Deeper floors hit harder, so they get a larger margin; values are tuned so a
# skilled policy wins a clear majority of seeds while reckless/trivial policies,
# which take many simultaneous hits, still die well short of the gate.
def _set_floor_nav_hp(env: CraftaxFullEnv, hp: int) -> None:
    env._max_hp = hp
    env._hp = hp


_FLOOR_GATE_TASK_TEXT = (
    "As in upstream Craftax, the stairs are sealed until you defeat "
    f"{_FLOOR_GATE_KILLS} hostile monsters on this floor; the HUD shows "
    "the kill count. "
)

# ===================================================================
# Helper: place a Craftax mob near a position
# ===================================================================

def _place_scenario_mob(
    env: CraftaxScenarioEnv,
    mob_type: str,
    near_x: int,
    near_y: int,
    radius: int = 4,
) -> None:
    """Place a mob of *mob_type* near (*near_x*, *near_y*) on walkable grass."""
    stats = _MOB_STATS[mob_type]
    for _att in range(60):
        dx = int(env.rng.integers(-radius, radius + 1))
        dy = int(env.rng.integers(-radius, radius + 1))
        x, y = near_x + dx, near_y + dy
        if (
            0 <= x < env._WORLD_SIZE
            and 0 <= y < env._WORLD_SIZE
            and env._world[y][x] in SURFACE_WALKABLE
            and (x, y) != (env._agent_x, env._agent_y)
            and not env._mob_at(x, y)
        ):
            mob: Mob = {
                "type": mob_type,
                "x": x,
                "y": y,
                "hp": stats["hp"],
                "max_hp": stats["hp"],
                "attack_cooldown": 0,
            }
            env._mobs.append(mob)
            return


def _place_full_mob(
    env: CraftaxFullEnv,
    mob_type: str,
    near_x: int,
    near_y: int,
    floor: int = 0,
    radius: int = 4,
    is_boss: bool = False,
    hp: int | None = None,
    max_hp: int | None = None,
) -> None:
    """Place a mob on the specified Craftax floor."""
    stats = FULL_MOB_STATS.get(mob_type, {"hp": 5, "damage": 2})
    actual_hp = hp if hp is not None else stats["hp"]
    actual_max = max_hp if max_hp is not None else stats["hp"]
    fsize = _SURFACE_SIZE if floor == 0 else _DUNGEON_SIZE
    grid = env._floors[floor]
    walkable = SURFACE_WALKABLE if floor == 0 else DUNGEON_WALKABLE
    for _att in range(80):
        dx = int(env.rng.integers(-radius, radius + 1))
        dy = int(env.rng.integers(-radius, radius + 1))
        x, y = near_x + dx, near_y + dy
        if (
            0 <= x < fsize
            and 0 <= y < fsize
            and grid[y][x] in walkable
            and (x, y) != (env._agent_x, env._agent_y)
            and not env._mob_at(x, y, floor)
        ):
            mob: FullMob = {
                "type": mob_type,
                "x": x,
                "y": y,
                "hp": actual_hp,
                "max_hp": actual_max,
                "is_boss": is_boss,
                "floor": floor,
                "attack_cooldown": 0,
            }
            env._mobs.append(mob)
            return


def _floor_transition_reward(
    env: CraftaxFullEnv,
    old_floor: int,
    target_floor: int,
    terminated: bool,
    info: dict[str, Any],
) -> tuple[float, bool]:
    """Pattern-A floor tasks: death beats success, timeout stays zero."""
    if env._hp <= 0:
        info["subtask_success"] = False
        return -1.0, True
    if env._current_floor == target_floor and old_floor != target_floor:
        info["subtask_success"] = True
        return 1.0, True
    return 0.0, terminated


def _is_floor_gate_monster(mob: FullMob) -> bool:
    from glyphbench.envs.craftax.mechanics.mobs import (
        MELEE_MOB_NAMES,
        RANGED_MOB_NAMES,
    )

    return (
        not bool(mob.get("is_boss", False))
        and mob.get("type") in MELEE_MOB_NAMES + RANGED_MOB_NAMES
    )


def _place_floor_gate_monster(
    env: CraftaxFullEnv,
    floor: int,
    mob_type: str,
    allowed_cells: set[tuple[int, int]] | None = None,
) -> bool:
    """Place an extra hostile floor monster for upstream-like stair gates."""
    stats = FULL_MOB_STATS.get(mob_type, {"hp": 5, "damage": 2})
    fsize = _SURFACE_SIZE if floor == 0 else _DUNGEON_SIZE
    grid = env._floors[floor]
    walkable = SURFACE_WALKABLE if floor == 0 else DUNGEON_WALKABLE
    forbidden = {
        env._stairs_up_pos.get(floor),
        env._stairs_down_pos.get(floor),
        (env._agent_x, env._agent_y),
    }

    def can_place(x: int, y: int) -> bool:
        return (
            0 <= x < fsize
            and 0 <= y < fsize
            and grid[y][x] in walkable
            and (allowed_cells is None or (x, y) in allowed_cells)
            and (x, y) not in forbidden
            and not env._mob_at(x, y, floor)
        )

    cells = (
        list(allowed_cells)
        if allowed_cells is not None
        else [(x, y) for y in range(1, fsize - 1) for x in range(1, fsize - 1)]
    )
    candidates = [(x, y) for x, y in cells if can_place(x, y)]
    if not candidates:
        return False
    x, y = candidates[int(env.rng.integers(0, len(candidates)))]

    env._mobs.append({
        "type": mob_type,
        "x": x,
        "y": y,
        "hp": stats["hp"],
        "max_hp": stats["hp"],
        "is_boss": False,
        "floor": floor,
        "attack_cooldown": 0,
    })
    return True


def _reachable_floor_cells(
    env: CraftaxFullEnv,
    floor: int,
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    grid = env._floors[floor]
    walkable = SURFACE_WALKABLE if floor == 0 else DUNGEON_WALKABLE
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= ny < len(grid)
                and 0 <= nx < len(grid[ny])
                and (nx, ny) not in seen
                and grid[ny][nx] in walkable
            ):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return seen


def _ensure_floor_route_connected(env: CraftaxFullEnv, floor: int) -> None:
    start = (env._agent_x, env._agent_y)
    stairs_down = env._stairs_down_pos.get(floor)
    if stairs_down is None:
        return
    if stairs_down in _reachable_floor_cells(env, floor, start):
        return

    grid = env._floors[floor]
    x, y = start
    tx, ty = stairs_down
    while x != tx:
        x += 1 if tx > x else -1
        if (x, y) != stairs_down:
            grid[y][x] = TILE_DUNGEON_FLOOR
    while y != ty:
        y += 1 if ty > y else -1
        if (x, y) != stairs_down:
            grid[y][x] = TILE_DUNGEON_FLOOR


def _ensure_floor_gate_monsters(env: CraftaxFullEnv, floor: int) -> None:
    from glyphbench.envs.craftax.mechanics.mobs import FLOOR_MOB_MAPPING

    _ensure_floor_route_connected(env, floor)
    mapping = FLOOR_MOB_MAPPING[floor]
    gate_types = (mapping["melee"], mapping["ranged"])
    reachable = _reachable_floor_cells(env, floor, (env._agent_x, env._agent_y))
    env._mobs = [
        mob for mob in env._mobs
        if not (
            int(mob.get("floor", -1)) == floor
            and _is_floor_gate_monster(mob)
            and (mob["x"], mob["y"]) not in reachable
        )
    ]
    current = sum(
        1
        for mob in env._mobs
        if int(mob.get("floor", -1)) == floor
        and _is_floor_gate_monster(mob)
        and (mob["x"], mob["y"]) in reachable
    )
    idx = 0
    while current < _FLOOR_GATE_KILLS:
        if not _place_floor_gate_monster(
            env,
            floor,
            gate_types[idx % len(gate_types)],
            allowed_cells=reachable,
        ):
            break
        current += 1
        idx += 1

    # Retune: thin the reachable gate-mob pool so a floor never swarms the
    # player far beyond what the (now 3-kill) gate requires. Worldgen can place
    # many high-damage mobs on the deeper realms; with the old 8-kill gate that
    # was the point, but a 3-kill gate over a 9-20 HP nav budget cannot survive
    # a 6-8-mob pile-on. We keep a small margin above the gate (choice of
    # targets) and defuse spawn-adjacent ambushes (the audited luck-sensitivity)
    # by preferentially removing the gate mobs closest to the spawn first.
    _cap_floor_gate_monsters(env, floor, reachable)


def _ensure_floor_gate_monsters_for(
    env: CraftaxFullEnv,
    floor: int,
    required: int,
    *,
    cap: bool = False,
) -> None:
    """Ensure a focused floor has at least ``required`` reachable gate mobs.

    The existing floor-nav tasks intentionally use the global 3-kill retune.
    P3 hard variants need local gates, including 8-kill single-floor drills,
    without changing that global setting or the older task contract.
    """
    from glyphbench.envs.craftax.mechanics.mobs import FLOOR_MOB_MAPPING

    _ensure_floor_route_connected(env, floor)
    mapping = FLOOR_MOB_MAPPING[floor]
    gate_types = (mapping["melee"], mapping["ranged"])
    start = (env._agent_x, env._agent_y)
    reachable = _reachable_floor_cells(env, floor, start)
    env._mobs = [
        mob for mob in env._mobs
        if not (
            int(mob.get("floor", -1)) == floor
            and _is_floor_gate_monster(mob)
            and (mob["x"], mob["y"]) not in reachable
        )
    ]
    current = sum(
        1
        for mob in env._mobs
        if int(mob.get("floor", -1)) == floor
        and _is_floor_gate_monster(mob)
        and (mob["x"], mob["y"]) in reachable
    )
    idx = 0
    while current < required:
        if not _place_floor_gate_monster(
            env,
            floor,
            gate_types[idx % len(gate_types)],
            allowed_cells=reachable,
        ):
            break
        current += 1
        idx += 1
    if cap:
        _cap_floor_gate_monsters(env, floor, reachable)


_FLOOR_GATE_MOB_CAP = _FLOOR_GATE_KILLS + 3
_FLOOR_GATE_SPAWN_SAFE_RADIUS = 3
_FLOOR_GATE_SPAWN_SAFE_MAX = 1


def _cap_floor_gate_monsters(
    env: CraftaxFullEnv,
    floor: int,
    reachable: set[tuple[int, int]],
) -> None:
    """Limit reachable gate mobs to a small margin above the kill gate and cap
    how many start within striking distance of the spawn (anti-pile-on)."""
    ax, ay = env._agent_x, env._agent_y

    def manh(mob: FullMob) -> int:
        return abs(mob["x"] - ax) + abs(mob["y"] - ay)

    gate_mobs = [
        mob for mob in env._mobs
        if int(mob.get("floor", -1)) == floor
        and _is_floor_gate_monster(mob)
        and (mob["x"], mob["y"]) in reachable
    ]
    if len(gate_mobs) <= _FLOOR_GATE_KILLS:
        return

    keep: list[FullMob] = []
    near_kept = 0
    # Sort farthest-first so the closest mobs are the first candidates for
    # removal when we exceed the spawn-safe quota or the overall cap.
    for mob in sorted(gate_mobs, key=manh, reverse=True):
        is_near = manh(mob) <= _FLOOR_GATE_SPAWN_SAFE_RADIUS
        if len(keep) >= _FLOOR_GATE_MOB_CAP:
            break
        if is_near and near_kept >= _FLOOR_GATE_SPAWN_SAFE_MAX:
            continue
        keep.append(mob)
        if is_near:
            near_kept += 1

    # Always retain at least the gate requirement worth of reachable mobs.
    if len(keep) < _FLOOR_GATE_KILLS:
        for mob in sorted(gate_mobs, key=manh):
            if mob not in keep:
                keep.append(mob)
            if len(keep) >= _FLOOR_GATE_KILLS:
                break

    keep_ids = {id(mob) for mob in keep}
    env._mobs = [
        mob for mob in env._mobs
        if not (
            int(mob.get("floor", -1)) == floor
            and _is_floor_gate_monster(mob)
            and (mob["x"], mob["y"]) in reachable
            and id(mob) not in keep_ids
        )
    ]


def _clear_area(grid: list[list[str]], cx: int, cy: int, r: int,
                size: int, tile: str = TILE_GRASS) -> None:
    """Set all tiles within radius *r* of (*cx*, *cy*) to *tile*."""
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < size and 0 <= y < size:
                grid[y][x] = tile


def _clear_dungeon_area(grid: list[list[str]], cx: int, cy: int, r: int,
                        size: int) -> None:
    """Set dungeon tiles to floor within radius."""
    _clear_area(grid, cx, cy, r, size, tile=TILE_DUNGEON_FLOOR)


def _scatter_scenario_tiles(
    env: CraftaxScenarioEnv,
    cx: int,
    cy: int,
    *,
    count: int,
    radius: int,
    min_manhattan: int,
    tiles: tuple[str, ...],
) -> None:
    """Scatter seed-varied terrain near the agent without blocking the task."""
    for _ in range(count):
        for _attempt in range(40):
            dx = int(env.rng.integers(-radius, radius + 1))
            dy = int(env.rng.integers(-radius, radius + 1))
            if abs(dx) + abs(dy) < min_manhattan:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < env._WORLD_SIZE and 0 <= y < env._WORLD_SIZE:
                env._world[y][x] = str(env.rng.choice(tiles))
                break


def _surface_reachable_with_dist(
    env: CraftaxScenarioEnv,
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    """BFS over SURFACE_WALKABLE from *start*, returning cell -> distance."""
    size = env._WORLD_SIZE
    queue: deque[tuple[int, int]] = deque([start])
    dist = {start: 0}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nc = (nx, ny)
            if (
                0 <= nx < size
                and 0 <= ny < size
                and nc not in dist
                and env._world[ny][nx] in SURFACE_WALKABLE
            ):
                dist[nc] = dist[(x, y)] + 1
                queue.append(nc)
    return dist


def _place_reachable_diamond(
    env: CraftaxScenarioEnv,
    start: tuple[int, int],
    min_dist: int,
    max_dist: int,
) -> tuple[int, int]:
    """Place one mineable diamond at a seed-randomized reachable location.

    The diamond is placed on a walkable cell whose BFS distance from *start*
    lies in [min_dist, max_dist] (so the agent must actually navigate around
    natural terrain — no fixed straight corridor). If no natural cell qualifies
    (small reachable region), a minimal winding path is carved toward a random
    direction. Returns the diamond's (x, y)."""
    dist = _surface_reachable_with_dist(env, start)
    sx, sy = start
    candidates = [
        cell for cell, d in dist.items()
        if min_dist <= d <= max_dist and cell != start
    ]
    if candidates:
        idx = int(env.rng.integers(0, len(candidates)))
        dx, dy = candidates[idx]
        env._world[dy][dx] = TILE_DIAMOND
        return (dx, dy)

    # Fallback (rare): carve a contiguous, slightly winding corridor of grass
    # one cell at a time toward a random heading, so the path stays connected
    # to the start, then drop the diamond at its far end. Steps alternate
    # between the two heading axes so it is not a perfectly straight line.
    primary, secondary = (1, 0), (0, 1)
    if int(env.rng.integers(0, 2)):
        primary, secondary = secondary, primary
    if int(env.rng.integers(0, 2)):
        primary = (-primary[0], -primary[1])
    if int(env.rng.integers(0, 2)):
        secondary = (-secondary[0], -secondary[1])
    target = int(env.rng.integers(min_dist, max_dist + 1))
    x, y = sx, sy
    for step in range(target):
        hx, hy = secondary if step % 3 == 2 else primary
        nx = max(1, min(env._WORLD_SIZE - 2, x + hx))
        ny = max(1, min(env._WORLD_SIZE - 2, y + hy))
        if (nx, ny) == (x, y):  # hit the border on this axis; use the other
            hx, hy = primary if (hx, hy) == secondary else secondary
            nx = max(1, min(env._WORLD_SIZE - 2, x + hx))
            ny = max(1, min(env._WORLD_SIZE - 2, y + hy))
        if env._world[ny][nx] != TILE_DIAMOND:
            env._world[ny][nx] = TILE_GRASS
        x, y = nx, ny
    env._world[y][x] = TILE_DIAMOND
    return (x, y)


def _carve_surface_route(
    env: CraftaxFullEnv,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    """Ensure a walkable surface route from *start* to *goal* exists; if the
    goal is not BFS-reachable, carve a grass corridor (L-shaped) connecting
    them. Operates on floor 0's grid."""
    grid = env._floors[0]
    size = len(grid)
    # BFS over current walkable surface.
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nc = (nx, ny)
            if (
                0 <= nx < size
                and 0 <= ny < size
                and nc not in seen
                and grid[ny][nx] in SURFACE_WALKABLE
            ):
                seen.add(nc)
                queue.append(nc)
    if goal in seen:
        return
    # Carve an L-shaped grass corridor (do not overwrite the staircase tile).
    x, y = start
    gx, gy = goal
    while x != gx:
        x += 1 if gx > x else -1
        if (x, y) != goal:
            grid[y][x] = TILE_GRASS
    while y != gy:
        y += 1 if gy > y else -1
        if (x, y) != goal:
            grid[y][x] = TILE_GRASS


def _learn_all_spells(env: CraftaxFullEnv) -> None:
    env._learned_spells = {"fireball": True, "iceball": True}


def _give_potions(env: CraftaxFullEnv, count: int = 1) -> None:
    env._inventory["potions"] = {
        color: count
        for color in ("red", "green", "blue", "pink", "cyan", "yellow")
    }


def _set_armor_enchants(env: CraftaxFullEnv, enchants: dict[str, int]) -> None:
    for slot, element in enchants.items():
        env._armor_enchants[slot] = element


class _FloorExitGateMixin:
    """Optional local-kill gate before descending from focused floor tasks."""

    _EXIT_KILL_REQUIREMENT = _FLOOR_GATE_KILLS
    _GATED_FLOORS: tuple[int, ...] = ()

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)  # type: ignore[misc]
        if self._EXIT_KILL_REQUIREMENT <= 0:
            self._GATED_FLOORS = ()
        self._floor_task_kills = {
            floor: 0 for floor in self._GATED_FLOORS
        }
        return obs

    def _record_floor_task_kill(self, mob: FullMob) -> None:
        floor = int(mob.get("floor", -1))
        if floor in self._GATED_FLOORS and _is_floor_gate_monster(mob):
            current = self._floor_task_kills.get(floor, 0)
            self._floor_task_kills[floor] = min(
                self._EXIT_KILL_REQUIREMENT,
                current + 1,
            )

    def _attack_mob(self, mob: FullMob) -> float:
        floor = int(mob.get("floor", -1))
        is_gate_monster = _is_floor_gate_monster(mob)
        reward = super()._attack_mob(mob)  # type: ignore[misc]
        if floor in self._GATED_FLOORS and is_gate_monster and mob not in self._mobs:
            self._record_floor_task_kill(mob)
        return reward

    def _attack_mob_kill(self, mob: FullMob) -> float:
        floor = int(mob.get("floor", -1))
        is_gate_monster = _is_floor_gate_monster(mob)
        reward = super()._attack_mob_kill(mob)  # type: ignore[misc]
        if floor in self._GATED_FLOORS and is_gate_monster:
            self._record_floor_task_kill(mob)
        return reward

    def _handle_descend(self) -> float:
        floor = self._current_floor
        grid = self._current_grid()
        if (
            self._EXIT_KILL_REQUIREMENT > 0
            and floor in self._GATED_FLOORS
            and grid[self._agent_y][self._agent_x] == TILE_STAIRS_DOWN
        ):
            kills = self._floor_task_kills.get(floor, 0)
            if kills < self._EXIT_KILL_REQUIREMENT:
                self._message = (
                    "Stairs sealed: defeat "
                    f"{self._EXIT_KILL_REQUIREMENT - kills} more hostile monsters "
                    f"({kills}/{self._EXIT_KILL_REQUIREMENT})."
                )
                return 0.0
        return super()._handle_descend()  # type: ignore[misc]

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()  # type: ignore[misc]
        floor = getattr(self, "_current_floor", -1)
        if (
            self._EXIT_KILL_REQUIREMENT <= 0
            or floor not in getattr(self, "_GATED_FLOORS", ())
        ):
            return obs

        kills = getattr(self, "_floor_task_kills", {}).get(floor, 0)
        state = "open" if kills >= self._EXIT_KILL_REQUIREMENT else "sealed"
        gate_line = (
            f"Stair seal: {state} "
            f"({kills}/{self._EXIT_KILL_REQUIREMENT} hostile monsters defeated)"
        )
        hud = f"{obs.hud}\n{gate_line}" if obs.hud else gate_line
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=hud,
            message=obs.message,
        )

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)  # type: ignore[misc]
        info["floor_task_kills"] = dict(getattr(self, "_floor_task_kills", {}))
        return obs, reward, terminated, truncated, info


_FLOOR_EXIT_ACTION_DISPATCH = dict(CraftaxFullEnv._ACTION_DISPATCH)
_FLOOR_EXIT_ACTION_DISPATCH["DESCEND"] = _FloorExitGateMixin._handle_descend
_FloorExitGateMixin._ACTION_DISPATCH = _FLOOR_EXIT_ACTION_DISPATCH


# ===================================================================
# Named dungeon-floor tasks (Craftax mechanics env)
# ===================================================================

class CraftaxDungeonEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Dungeon. Start with stone sword + stone pickaxe.
    Goal: reach the stairs down. Combat is optional. Max 480 steps."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "floors:1", "floors:2", "floors:navigation",
    )

    _GATED_FLOORS = (1,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-dungeon-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Dungeon (floor 1). Find the stairs down (⇣) and "
            "use DESCEND to reach the Gnomish Mines. "
            f"{_FLOOR_GATE_TASK_TEXT}"
            "You start with a stone sword and stone pickaxe. Snails, orc "
            "soldiers, and orc mages patrol the rooms; avoid or fight only "
            "when it helps navigation. Reward: +1 for descending, -1 on "
            "death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Teleport player to floor 1
        self._current_floor = 1
        up_pos = self._stairs_up_pos.get(1)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        # Give starting gear
        self._inventory = {
            "stone_sword": 1,
            "stone_pickaxe": 1,
            "wood": 3,
            "coal": 3,
        }
        _set_floor_nav_hp(self, 20)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        _ensure_floor_gate_monsters(self, 1)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 2, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxGnomishMinesEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Gnomish Mines. Start with iron sword + iron armor.
    Goal: find stairs down. Combat is optional. Max 220 steps."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "floors:2", "floors:3", "floors:navigation",
    )

    _GATED_FLOORS = (2,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-gnomish-mines-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Gnomish Mines (floor 2). Find the stairs down (⇣) "
            "and use DESCEND to reach the Sewers. You start with an "
            f"iron sword and full iron armor. {_FLOOR_GATE_TASK_TEXT}"
            "Mine corridors contain gems, ore "
            "pockets, bats, gnome warriors, and gnome archers; combat is "
            "optional unless it blocks the route. Reward: +1 for descending, "
            "-1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 2
        up_pos = self._stairs_up_pos.get(2)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "iron_sword": 1,
            "wood": 5,
            "coal": 5,
        }
        # Phase γ T03γ: armour tracked in _armor_slots (4-slot dict).
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1  # all iron tier
        _set_floor_nav_hp(self, 20)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        _ensure_floor_gate_monsters(self, 2)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 3, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxSewersEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Sewers. Start with iron sword + iron armor + spells.
    Goal: find stairs down. Combat is optional. Max 240 steps."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "survival:mana", "magic:spells", "magic:enchants", "items:potions",
        "floors:3", "floors:4", "floors:navigation",
    )

    _GATED_FLOORS = (3,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-sewers-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Sewers (floor 3, home of the ice enchant table). "
            "Find the stairs down (⇣) and use DESCEND to reach the Vaults, "
            f"using combat when needed. {_FLOOR_GATE_TASK_TEXT}"
            "You have iron gear, "
            "learned spells, and potions. Reward: +1 for descending, -1 on "
            "death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 3
        up_pos = self._stairs_up_pos.get(3)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "iron_sword": 1,
            "wood": 5,
            "coal": 5,
        }
        # Phase γ T03γ: armour tracked in _armor_slots (4-slot dict).
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1  # all iron tier
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_floor_nav_hp(self, 24)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        _ensure_floor_gate_monsters(self, 3)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 4, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxBossFightEnv(CraftaxFullEnv):
    """Final boss fight. Start with diamond gear on a boss floor.
    Goal: defeat the boss. Max 480 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "legend:mobs:boss", "items:bow", "floors:5", "floors:navigation",
    )

    # Pattern D: +1.0 on boss kill, -1.0 on death.
    _DEATH_PENALTY = -1.0
    _GATED_FLOORS = (5,)
    _emit_death_penalty: bool = False

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-bossfight-v0"

    def _task_description(self) -> str:
        return (
            "Defeat the lich boss on dungeon floor 5. You start on floor 5 "
            "with a diamond sword (fire-enchanted), full diamond armor, both "
            "spells learned, and potions. Use DO facing the lich to "
            "melee or CAST_FIREBALL for ranged damage. The lich also has a "
            "ranged attack within 4 tiles (non-projectile), so keeping "
            "distance does not always make you safe. It is rendered as W; "
            "B is a walkable boss door that blocks projectiles. Reward: "
            "+1 on kill, -1 on death, 0 on timeout; episode ends. Time "
            "limit: 480 steps. (Note: this is the lich, not the necromancer "
            "— see craftax-necromancer-v0 for the floor-8 final boss.)"
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Go to floor 5 (the deepest with the lich boss)
        self._current_floor = 5
        up_pos = self._stairs_up_pos.get(5)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "wood": 3,
            "coal": 3,
        }
        self._sword_enchantment = 1  # fire-enchanted by default (phase γ T10γ)
        # Phase γ T03γ: armour tracked in _armor_slots (4-slot dict).
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # all diamond tier
        _learn_all_spells(self)
        _give_potions(self, count=1)
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        # Track initial boss count
        self._boss_count_at_start = sum(
            1 for m in self._mobs
            if m["is_boss"] and m["floor"] == 5
        )
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Replace any parent-emitted reward with structural Pattern D shape:
        # +1 on boss kill, -1 on death, additive. A kamikaze swing that
        # drops the boss and the player on the same tick yields +1 + -1 = 0
        # — informative (agent did the kill, but cannot claim survival).
        # Cumulative return stays in [-1, +1].
        reward = 0.0
        boss_killed_this_episode = not self._bosses_alive.get(5, True)
        if boss_killed_this_episode:
            # Reward only fires once because terminated=True the same tick;
            # if the boss dies but the player also dies, success flag below
            # is overridden to False (death dominates the success label).
            reward += 1.0
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


# ===================================================================
# 5-8: Survival Tasks (surface scenario)
# ===================================================================

class CraftaxSurviveHungerEnv(CraftaxScenarioEnv):
    """Start with food=2, no food nearby. Must find food.
    Pattern C: +1/2 on first food source and +1/2 on surviving 120 steps
    with food eaten; -1 on death."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)
        self._food_eaten: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-hunger-v0"

    def _task_description(self) -> str:
        return (
            "You start starving (food=1/9). Find food before you starve and "
            "survive 120 steps. Kill a cow with DO for food, or eat a ripe "
            "plant with EAT_PLANT. Food drains 1 every 50 steps; at 0 food "
            "you take -1 HP/step. You have a wood sword. Reward: +1/2 "
            "when you first eat food and +1/2 if you survive to step 120 "
            "AND ate at least one food source; -1 if you die."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Food=1 so the first drain at step 50 hits 0; damage starts at
        # step 51 and the HP-9 agent dies at step 60 if no food is
        # found. Forces real food acquisition before max_turns=120.
        self._food = 1
        self._food_eaten = 0
        # Remove nearby food sources (ripe plants)
        cx, cy = self._agent_x, self._agent_y
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                    and self._world[y][x] == TILE_RIPE_PLANT
                ):
                    self._world[y][x] = TILE_GRASS
        self._inventory = {"wood_sword": 1}
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_food = self._food
        obs, reward, terminated, truncated, info = super()._step(action)
        # Suppress parent reward, then credit first-food milestone even on
        # the death tick so training signal preserves "almost made it".
        reward = 0.0
        if self._food > old_food:
            if self._food_eaten == 0:
                reward += 0.5
            self._food_eaten += 1
        # Apply -1 death penalty additively (never as an override that
        # would discard the first-food milestone earned same tick).
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        elif self._turn >= self.max_turns and self._food_eaten >= 1:
            reward += 0.5
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSurviveThirstEnv(CraftaxScenarioEnv):
    """Start with water=2. Must find water source. Max 120 steps.
    Pattern C: +1/2 on first drink and +1/2 on surviving with water
    drunk; -1 on death."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "legend:projectiles",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee", "combat:projectiles",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)
        self._drinks: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-thirst-v0"

    def _task_description(self) -> str:
        return (
            "You start dehydrated (water=2/9). Find a water tile and use "
            "DRINK_WATER facing it to drink (+1 water). Nearby water within "
            "6 tiles has been removed, but a reachable water source is seeded "
            "outside that start area. Survive 120 steps without dying of "
            "thirst. Water drains 1 every 40 steps; at 0 water you take "
            "-1 HP/step. Reward: +1/2 when you first drink water and +1/2 "
            "if you survive to step 120 AND drank at least once "
            "(NOOP-spam can't game it); -1 if you die."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._water = 2
        self._drinks = 0
        # Remove water tiles in immediate vicinity to force exploration
        cx, cy = self._agent_x, self._agent_y
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                    and self._world[y][x] == TILE_WATER
                ):
                    self._world[y][x] = TILE_GRASS
        self._place_reachable_water(cx, cy)
        self._inventory = {}
        return self._render_current_observation()

    def _place_reachable_water(self, cx: int, cy: int) -> None:
        """Guarantee one water source just outside the forced-search radius."""
        offsets = [
            (10, 0), (-10, 0), (0, 10), (0, -10),
            (8, 6), (8, -6), (-8, 6), (-8, -6),
        ]
        start = int(self.rng.integers(0, len(offsets)))
        for i in range(len(offsets)):
            dx, dy = offsets[(start + i) % len(offsets)]
            x, y = cx + dx, cy + dy
            if 1 <= x < self._WORLD_SIZE - 1 and 1 <= y < self._WORLD_SIZE - 1:
                self._world[y][x] = TILE_WATER
                return

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        action_name = self.action_spec.names[action]
        drank_from_water = False
        if action_name == "DRINK_WATER":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            drank_from_water = (
                0 <= fx < self._WORLD_SIZE
                and 0 <= fy < self._WORLD_SIZE
                and self._world[fy][fx] == TILE_WATER
            )
        old_water = self._water
        obs, reward, terminated, truncated, info = super()._step(action)
        # Suppress parent reward, then credit first-drink milestone even on
        # the death tick so training signal preserves "almost made it".
        reward = 0.0
        if drank_from_water or self._water > old_water:
            if self._drinks == 0:
                reward += 0.5
            self._drinks += 1
        # Apply -1 death penalty additively.
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        elif self._turn >= self.max_turns and self._drinks >= 1:
            reward += 0.5
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSurviveNightEnv(CraftaxScenarioEnv):
    """Start at dusk. Survive the night with basic gear. Max 150 steps."""

    _EXTRA_WAVE_TURNS = frozenset({25, 50, 75})

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "legend:projectiles",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee", "combat:projectiles",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-night-v0"

    def _task_description(self) -> str:
        return (
            "Night is falling — monsters will spawn. Survive until dawn. You "
            "have a stone sword and only 4 stone blocks to build shelter. "
            "Extra zombies and skeletons arrive during the night. Place stone "
            "blocks around yourself for protection, or fight monsters with DO. "
            "Reward: +1 if alive when the sun rises; -1 if you die during "
            "the night."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Set time to just before nightfall
        self._day_counter = _DAY_LENGTH - 2
        self._day_night = "day"
        self._inventory = {
            "stone_sword": 1,
            "stone": 4,
        }
        self._hp = 8
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._survived_night = False
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_phase = self._day_night
        obs, reward, terminated, truncated, info = super()._step(action)
        if (
            not terminated
            and self._day_night == "night"
            and self._turn in self._EXTRA_WAVE_TURNS
        ):
            mob_type = "skeleton" if self._turn == 50 else "zombie"
            _place_scenario_mob(self, mob_type, self._agent_x, self._agent_y, radius=6)
        # Pattern A: +1.0 on terminal "survived through dawn" event. The -1
        # death penalty is applied additively (no per-step shaping here, so
        # this is equivalent to override; written additively for consistency
        # with the rest of the suite).
        reward = 0.0
        if old_phase == "night" and self._day_night == "day" and self._hp > 0:
            reward = 1.0
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxSurviveWildEnv(CraftaxScenarioEnv):
    """Start with nothing, day 1. Survive 200 steps managing all needs."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "legend:projectiles",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee", "combat:projectiles",
        "crafting:wood", "crafting:stone",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-wild-v0"

    # Pattern A: +0.1 per 20-step milestone survived. This keeps the same
    # cumulative +1.0 cap while avoiding invisible +1/200 tick rewards.
    _SURVIVE_TARGET = 200
    _SURVIVE_MILESTONE_STEPS = 20
    _SURVIVE_MILESTONES = 10

    def _task_description(self) -> str:
        return (
            "You start with nothing. Survive 200 steps managing hunger, "
            "thirst, energy, and nighttime monsters. This focused start sets "
            "Food and Water to 4/9 and begins near nightfall, so shelter, "
            "food, and water matter immediately. Chop trees (DO) for "
            "wood, craft tools at a table, find water (≈) to DRINK_WATER, "
            "kill cows or eat plants for food, and build shelter before night. "
            "Reward: +0.1 at each 20-step survival milestone "
            "(sum to +1.0 on full survival); "
            "-1 if you die before time runs out."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._inventory = {}
        self._hp = 9
        # Start food/water below max so the survival drains actually
        # bite during a 200-step episode (full pools survive 200 steps
        # of NOOP otherwise).
        self._food = 4
        self._water = 4
        self._energy = _MAX_ENERGY
        # Begin near nightfall so the day/night transition happens mid
        # episode and forces engagement with shelter/sleep/combat.
        # _DAY_LENGTH=200 means day phase runs 0..199; setting counter
        # to 170 puts nightfall at step ~30 of the episode.
        self._day_counter = 170
        self._survive_milestones_paid = 0
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Pattern A: +0.1 per 20-step milestone survived.
        reward = 0.0
        if not terminated:
            milestones = min(
                self._SURVIVE_MILESTONES,
                self._turn // self._SURVIVE_MILESTONE_STEPS,
            )
            newly_paid = milestones - self._survive_milestones_paid
            if newly_paid > 0:
                reward = newly_paid / self._SURVIVE_MILESTONES
                self._survive_milestones_paid = milestones
        # Apply -1 death penalty additively so cumulative return is
        # (milestones_survived / 10) - 1, not a clean -1 that loses progress.
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        elif (truncated or self._turn >= self.max_turns) and self._hp > 0:
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 9-14: Combat Tasks per Mob Type
# ===================================================================

class CraftaxFightCowEnv(CraftaxScenarioEnv):
    """Track down a cow in a seed-varied meadow; a wolf joins at step 30."""

    # Wolf (zombie-skinned) appears at this turn to add real combat pressure.
    _WOLF_SPAWN_TURN = 30
    _WOLF_KILL_REWARD = 0.5

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 80) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-cow-v0"

    def _task_description(self) -> str:
        return (
            "Track down a cow in a seed-generated meadow and defeat it for "
            "food by facing it and using DO to attack. You have a wood sword "
            "(+1 damage). At step 30 a wolf (z) appears and chases you. "
            "Reward is capped at +1: killing the wolf gives +0.5 partial "
            "credit but does not end the episode; defeating the cow gives "
            "the remaining credit and completes the task."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Remove all existing mobs
        self._mobs = []
        # Place exactly 1 cow far enough away that this is a hunt, not a
        # one-action attack.
        cx, cy = self._agent_x, self._agent_y
        stats = _MOB_STATS["cow"]
        for _attempt in range(120):
            dx = int(self.rng.integers(-9, 10))
            dy = int(self.rng.integers(-9, 10))
            if abs(dx) + abs(dy) < 7:
                continue
            x, y = cx + dx, cy + dy
            if (
                0 <= x < self._WORLD_SIZE
                and 0 <= y < self._WORLD_SIZE
                and self._world[y][x] in SURFACE_WALKABLE
            ):
                self._mobs.append({
                    "type": "cow",
                    "x": x,
                    "y": y,
                    "hp": stats["hp"],
                    "max_hp": stats["hp"],
                    "attack_cooldown": 0,
                })
                break
        self._inventory = {"wood_sword": 1}
        self._hp = 9
        self._wolf_spawned = False
        self._wolf_credit_given = False
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _spawn_wolf(self) -> None:
        """Spawn a single chasing zombie ('wolf') near the agent."""
        zombie_stats = _MOB_STATS["zombie"]
        cx, cy = self._agent_x, self._agent_y
        for _attempt in range(120):
            dx = int(self.rng.integers(-7, 8))
            dy = int(self.rng.integers(-7, 8))
            if abs(dx) + abs(dy) < 4:
                continue
            x, y = cx + dx, cy + dy
            if (
                0 <= x < self._WORLD_SIZE
                and 0 <= y < self._WORLD_SIZE
                and self._world[y][x] in SURFACE_WALKABLE
                and not self._mob_at(x, y)
            ):
                self._mobs.append({
                    "type": "zombie",
                    "x": x,
                    "y": y,
                    "hp": zombie_stats["hp"],
                    "max_hp": zombie_stats["hp"],
                    "attack_cooldown": 0,
                })
                self._wolf_spawned = True
                return

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Spawn wolf at step 30 (before parent step so AI can react).
        if (
            not self._wolf_spawned
            and self._turn >= self._WOLF_SPAWN_TURN
        ):
            self._spawn_wolf()
        cow_before = sum(1 for m in self._mobs if m["type"] == "cow")
        wolf_before = sum(1 for m in self._mobs if m["type"] == "zombie")
        obs, reward, terminated, truncated, info = super()._step(action)
        cow_after = sum(1 for m in self._mobs if m["type"] == "cow")
        wolf_after = sum(1 for m in self._mobs if m["type"] == "zombie")
        # Pattern C: ignore parent reward; terminal +1 on cow kill (full),
        # +0.5 on wolf kill (partial credit if cow unreachable). Death is
        # applied additively so a "kill + die same tick" preserves the kill
        # credit (final reward = kill_reward - 1, not a clean -1).
        reward = 0.0
        if cow_after < cow_before:
            reward += 0.5 if self._wolf_credit_given else 1.0
            terminated = True
            info["subtask_success"] = True
        elif wolf_after < wolf_before and not self._wolf_credit_given:
            reward += self._WOLF_KILL_REWARD
            self._wolf_credit_given = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFightZombiesEnv(CraftaxScenarioEnv):
    """Start with stone sword. 5 zombies in arena. Kill all. Max 130 steps."""

    # Pattern A: +1/N per zombie killed (sums to +1.0 on full clear).
    _KILL_TARGET = 5
    # Manage death penalty in own _step so partial kill rewards don't stack
    # with the parent's -1 (would yield -0.66 instead of -1).
    _emit_death_penalty: bool = False
    _disable_survival: bool = True
    _disable_day_night: bool = True

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 130) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-zombies-v0"

    def _task_description(self) -> str:
        return (
            "5 zombies (z) are closing in. Defeat all of them. Face a zombie "
            "and use DO to attack. You have a stone sword (+2 damage per hit). "
            "Each zombie has 5 HP and deals 2 damage when adjacent (so two "
            "hits per zombie). Reward: +1/5 per zombie killed (sums to +1.0 "
            "when all are dead), -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_scenario_mob(
                self, "zombie", self._agent_x, self._agent_y, radius=5
            )
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        # Disable survival drain for focused combat
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._zombie_kills_credited = 0
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        zombie_count_before = sum(
            1 for m in self._mobs if m["type"] == "zombie"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        zombie_count_after = sum(
            1 for m in self._mobs if m["type"] == "zombie"
        )
        kills = max(0, zombie_count_before - zombie_count_after)
        # Replace parent reward; the parent already suppressed its own -1
        # death penalty (_emit_death_penalty=False), so we apply death
        # additively below. Kill credit is preserved on every step,
        # including the final kill-and-die tick.
        previous_kills = self._zombie_kills_credited
        self._zombie_kills_credited = min(
            self._KILL_TARGET,
            self._zombie_kills_credited + kills,
        )
        reward = (self._zombie_kills_credited - previous_kills) / self._KILL_TARGET
        if self._zombie_kills_credited >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFightSkeletonsEnv(CraftaxScenarioEnv):
    """Start with wood sword. 4 skeletons. Must close distance. Max 150 steps."""

    # Pattern A: +1/N per skeleton killed (sums to +1.0 on full clear).
    _KILL_TARGET = 4
    # Manage death penalty in own _step so partial kill rewards don't stack
    # with the parent's -1 (would yield -0.66 instead of -1).
    _emit_death_penalty: bool = False
    _disable_survival: bool = True
    _disable_day_night: bool = True

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-skeletons-v0"

    def _task_description(self) -> str:
        return (
            "4 skeletons (k) are prowling nearby. Defeat all 4. Each skeleton "
            "has 3 HP and fires 2-damage arrows. You have a wood sword, so "
            "each skeleton takes multiple melee hits. Close distance, avoid "
            "arrows, and use DO to attack. Reward: +1/4 per skeleton "
            "killed (sums to +1.0 when all are dead), -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_scenario_mob(
                self, "skeleton", self._agent_x, self._agent_y, radius=6
            )
        self._inventory = {"wood_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._skeleton_kills_credited = 0
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        skel_count_before = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        skel_count_after = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        kills = max(0, skel_count_before - skel_count_after)
        # Replace parent reward; the parent already suppressed its own -1
        # death penalty (_emit_death_penalty=False), so we apply death
        # additively below. Kill credit is preserved on every step.
        previous_kills = self._skeleton_kills_credited
        self._skeleton_kills_credited = min(
            self._KILL_TARGET,
            self._skeleton_kills_credited + kills,
        )
        reward = (
            self._skeleton_kills_credited - previous_kills
        ) / self._KILL_TARGET
        if self._skeleton_kills_credited >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFightArchersEnv(CraftaxFullEnv):
    """Start with limited bow kit. 5 skeletons at range. Max 160 steps."""

    # Pattern D: 5 mobs to kill. +1/N per kill. -1.0 on death.
    _KILL_TARGET = 5
    _DEATH_PENALTY = -1.0
    # Suppress despawn on floor 0 so "run 14 cells away" can't despawn the
    # targets and falsely credit them as kills via mob-count delta.
    _GATED_FLOORS = (0,)

    tutorial_sections = (
        "overview",
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items", "legend:projectiles", "legend:hud",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
        "combat:melee", "combat:ranged_player", "combat:ranged_mob",
        "combat:armor", "combat:projectiles",
        "magic:spells", "magic:books", "magic:enchants",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement", "crafting:arrows",
        "items:resources", "items:bow", "items:potions",
        "progression:xp", "progression:attributes",
        "floors:0", "floors:navigation",
    )

    def __init__(self, max_turns: int = 160) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-archers-v0"

    def _task_description(self) -> str:
        return (
            "5 skeletons (a) are attacking from range. Defeat all 5. Each has "
            "3 HP and fires 2-damage arrows. You have a stone sword, a bow "
            "with 4 arrows, and enough wood+stone to craft a few more arrows "
            "at the nearby table. No spells or potion bailout are provided. "
            "Line up shots, craft arrows when needed, or close distance for "
            "DO melee. Reward: +1/5 per skeleton killed, -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Set up an arena on the surface
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 8, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        # Remove all mobs, then place skeletons (upstream ranged)
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(
                self, "skeleton", cx, cy, floor=0, radius=7
            )
        self._floors[0][cy][cx + 2] = TILE_TABLE
        self._inventory = {
            "stone_sword": 1,
            "bow": 1,
            "arrows": 4,
            "wood": 3,
            "stone": 3,
        }
        self._learned_spells = {"fireball": False, "iceball": False}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    # Suppress parent achievement and boss-kill rewards.
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        archer_count_before = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        # Drop the literal +10 boss-kill bonus emitted by CraftaxFullEnv
        # (none here, but defensive: any large parent reward is unwanted).
        archer_count_after = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        kills = archer_count_before - archer_count_after
        # Replace parent reward with structural per-step shaping.
        reward = kills * (1.0 / self._KILL_TARGET)
        if archer_count_after == 0 and archer_count_before > 0:
            terminated = True
            info["subtask_success"] = True
        # Death penalty applied additively so the final kill still credits
        # even if the killing blow also killed the player. subtask_success
        # is False on death (death dominates the success label).
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFightSpidersEnv(CraftaxFullEnv):
    """Start with iron sword. 3 kobolds. Max 120 steps.

    The registered env ID is preserved for compatibility; the scenario uses
    upstream kobolds (ranged, throws daggers).
    """

    # Pattern D: 3 kobolds. +1/N per kill. -1.0 on death.
    _KILL_TARGET = 3
    _DEATH_PENALTY = -1.0
    # Suppress despawn on floor 0 so the count-delta kill metric can't be
    # gamed by running 14 cells away.
    _GATED_FLOORS = (0,)

    tutorial_sections = (
        "overview",
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items", "legend:projectiles", "legend:hud",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
        "combat:melee", "combat:ranged_player", "combat:ranged_mob",
        "combat:armor", "combat:projectiles",
        "magic:spells", "magic:books", "magic:enchants",
        "crafting:wood", "crafting:stone", "crafting:iron", "crafting:diamond",
        "crafting:placement", "crafting:arrows",
        "items:resources", "items:bow", "items:potions",
        "progression:xp", "progression:attributes",
        "floors:0", "floors:navigation",
    )

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-spiders-v0"

    def _task_description(self) -> str:
        return (
            "3 kobolds (q) lurk nearby and throw daggers. Defeat all 3. Each "
            "has 8 HP and throws physical daggers. You have an iron sword (+3 damage) — "
            "use DO when adjacent. Reward: +1/3 per kobold killed, -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 6, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        self._mobs = []
        for _ in range(3):
            _place_full_mob(self, "kobold", cx, cy, floor=0, radius=4)
        self._inventory = {"iron_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        kobold_count_before = sum(
            1 for m in self._mobs if m["type"] == "kobold"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        kobold_count_after = sum(
            1 for m in self._mobs if m["type"] == "kobold"
        )
        kills = kobold_count_before - kobold_count_after
        reward = kills * (1.0 / self._KILL_TARGET)
        if kobold_count_after == 0 and kobold_count_before > 0:
            terminated = True
            info["subtask_success"] = True
        # Death penalty applied additively so the final kill still credits.
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFightBatsEnv(CraftaxFullEnv):
    """Start with stone sword. 6 bats plus ranged pressure. Max 130 steps."""

    # Pattern D: 6 bats. +1/N per kill. -1.0 on death.
    _KILL_TARGET = 6
    _DEATH_PENALTY = -1.0
    # Suppress despawn on floor 0 so the bat-count delta is a true kill
    # count, not "I walked 14 cells away and the bats disappeared".
    _GATED_FLOORS = (0,)

    tutorial_sections = (
        "overview",
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items", "legend:projectiles", "legend:hud",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
        "combat:melee", "combat:ranged_player", "combat:ranged_mob",
        "combat:armor", "combat:projectiles",
        "magic:spells", "magic:books", "magic:enchants",
        "crafting:wood", "crafting:stone", "crafting:iron", "crafting:diamond",
        "crafting:placement", "crafting:arrows",
        "items:resources", "items:bow", "items:potions",
        "progression:xp", "progression:attributes",
        "floors:0", "floors:navigation",
    )

    def __init__(self, max_turns: int = 130) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-bats-v0"

    def _task_description(self) -> str:
        return (
            "6 bats (b) are swarming you while 2 skeletons pressure you from "
            "range. Defeat all 6 bats. Each bat has 4 HP and moves "
            "erratically. You have a stone sword. Skeleton kills give no "
            "reward, but ignoring them is dangerous. Reward: +1/6 per bat "
            "killed, -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 5, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "bat", cx, cy, floor=0, radius=3)
        for _ in range(2):
            _place_full_mob(self, "skeleton", cx, cy, floor=0, radius=6)
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        bat_count_before = sum(
            1 for m in self._mobs if m["type"] == "bat"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        bat_count_after = sum(
            1 for m in self._mobs if m["type"] == "bat"
        )
        kills = bat_count_before - bat_count_after
        reward = kills * (1.0 / self._KILL_TARGET)
        if bat_count_after == 0 and bat_count_before > 0:
            terminated = True
            info["subtask_success"] = True
        # Death penalty applied additively so the final kill still credits.
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


# ===================================================================
# Magic, ranged, enchantment, and potion tasks
# ===================================================================

class CraftaxLearnSpellEnv(CraftaxFullEnv):
    """Open a dungeon chest, read the book, then cast the learned spell."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "survival:mana", "magic:books", "magic:spells",
        "items:potions", "floors:3", "floors:navigation",
    )

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-learn-spell-v0"

    def _task_description(self) -> str:
        return (
            "Find and open the chest in the Sewers, read the book it grants, "
            "then cast the newly learned fireball or iceball once. The "
            "chest is seed-varied and not adjacent to spawn. Reward: +1/3 "
            "for opening the chest, +1/3 for learning a spell, +1/3 for "
            "casting a learned spell; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 3
        up_pos = self._stairs_up_pos.get(3)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        self._inventory = {"iron_sword": 1}
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        self._learned_spells = {"fireball": False, "iceball": False}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        self._spell_task_opened_chest = False
        self._spell_task_learned = False
        self._spell_task_cast = False

        grid = self._floors[3]
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if abs(x - self._agent_x) + abs(y - self._agent_y) <= 8 and cell == TILE_CHEST:
                    grid[y][x] = TILE_DUNGEON_FLOOR
        for _attempt in range(100):
            dx = int(self.rng.integers(-12, 13))
            dy = int(self.rng.integers(-12, 13))
            if abs(dx) + abs(dy) < 10:
                continue
            x, y = self._agent_x + dx, self._agent_y + dy
            if (
                1 <= x < _DUNGEON_SIZE - 1
                and 1 <= y < _DUNGEON_SIZE - 1
                and grid[y][x] == TILE_DUNGEON_FLOOR
            ):
                grid[y][x] = TILE_CHEST
                break
        return self._render_current_observation()

    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_books = self._inventory.get("book", 0)
        old_known = sum(1 for known in self._learned_spells.values() if known)
        old_projectiles = len(self._player_projectiles)
        obs, reward, terminated, truncated, info = super()._step(action)
        # Always credit milestones first; the -1 death penalty is applied
        # additively below so progress earned on the death tick survives
        # into the cumulative return.
        reward = 0.0
        if not self._spell_task_opened_chest and self._inventory.get("book", 0) > old_books:
            self._spell_task_opened_chest = True
            reward += 1.0 / 3.0
        known = sum(1 for learned in self._learned_spells.values() if learned)
        if not self._spell_task_learned and known > old_known:
            self._spell_task_learned = True
            reward += 1.0 / 3.0
        if (
            not self._spell_task_cast
            and len(self._player_projectiles) > old_projectiles
            and any(self._learned_spells.values())
        ):
            self._spell_task_cast = True
            reward += 1.0 / 3.0
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info


class CraftaxEnchantWeaponEnv(CraftaxFullEnv):
    """Enchant a weapon, then use it in a small dungeon fight."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "survival:mana", "combat:elemental", "magic:enchants",
        "items:gems", "items:potions", "floors:4", "floors:navigation",
    )

    _KILL_TARGET = 2

    def __init__(self, max_turns: int = 140) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-enchant-weapon-v0"

    def _task_description(self) -> str:
        return (
            "In a curated Vaults setup, find the nearby fire enchant table, "
            "enchant your pre-supplied diamond sword with ENCHANT_WEAPON using "
            "ruby and mana, then defeat 2 nearby kobolds. This isolates the "
            "enchant mechanic from the full upstream gear grind. Reward: +1/2 "
            "for enchanting and +1/4 per kill, -1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 4
        up_pos = self._stairs_up_pos.get(4)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        self._inventory = {"diamond_sword": 1, "ruby": 1}
        self._sword_enchantment = 0
        self._mana = _MAX_MANA
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._enchant_task_done = False
        self._enchant_task_kills = 0
        self._mobs = [m for m in self._mobs if m["floor"] != 4 or m["is_boss"]]

        grid = self._floors[4]
        _clear_dungeon_area(grid, self._agent_x, self._agent_y, 4, _DUNGEON_SIZE)
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        start = seed % len(dirs)
        for i in range(len(dirs)):
            dx, dy = dirs[(start + i) % len(dirs)]
            table_x = self._agent_x + 2 * dx
            table_y = self._agent_y + 2 * dy
            stand_x = self._agent_x + dx
            stand_y = self._agent_y + dy
            if (
                1 <= table_x < _DUNGEON_SIZE - 1
                and 1 <= table_y < _DUNGEON_SIZE - 1
                and 1 <= stand_x < _DUNGEON_SIZE - 1
                and 1 <= stand_y < _DUNGEON_SIZE - 1
            ):
                grid[stand_y][stand_x] = TILE_DUNGEON_FLOOR
                grid[table_y][table_x] = TILE_ENCHANT_FIRE
                self._facing = (dx, dy)
                break
        kobold_stats = FULL_MOB_STATS["kobold"]
        mob_offsets = [(3, 0), (0, 3), (-3, 0), (0, -3)]
        placed = 0
        for dx, dy in mob_offsets:
            if placed >= self._KILL_TARGET:
                break
            x = self._agent_x + dx
            y = self._agent_y + dy
            if (
                1 <= x < _DUNGEON_SIZE - 1
                and 1 <= y < _DUNGEON_SIZE - 1
                and grid[y][x] == TILE_DUNGEON_FLOOR
                and not self._mob_at(x, y, 4)
            ):
                self._mobs.append({
                    "type": "kobold",
                    "x": x,
                    "y": y,
                    "hp": kobold_stats["hp"],
                    "max_hp": kobold_stats["hp"],
                    "is_boss": False,
                    "floor": 4,
                    "attack_cooldown": 0,
                })
                placed += 1
        return self._render_current_observation()

    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        before = sum(1 for m in self._mobs if m["floor"] == 4 and m["type"] == "kobold")
        obs, reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if not self._enchant_task_done and self._sword_enchantment != 0:
            self._enchant_task_done = True
            reward += 0.5
        after = sum(1 for m in self._mobs if m["floor"] == 4 and m["type"] == "kobold")
        kills = before - after
        if kills > 0:
            self._enchant_task_kills += kills
            reward += kills * (0.5 / self._KILL_TARGET)
        if self._enchant_task_done and self._enchant_task_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        # Apply -1 death penalty additively (preserves enchant + kill credit).
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxBowKiteEnv(CraftaxFullEnv):
    """Use bow spacing and arrows to clear ranged mobs."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS

    _KILL_TARGET = 4
    _GATED_FLOORS = (0,)  # consumed by CraftaxFullEnv despawn suppression

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-bow-kite-v0"

    def _task_description(self) -> str:
        return (
            "You have a bow with 6 arrows, wood+stone for MAKE_ARROW, and a "
            "wood sword as fallback. Keep distance, line up shots with "
            "SHOOT_ARROW, craft arrows at the nearby table when ammo runs "
            "low, and defeat 4 ranged mobs. These target mobs are pinned for "
            "the focused fight and do not despawn when far away. Reward: "
            "+1/4 per kill, -1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 0
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 6, _SURFACE_SIZE)
        # Halve tree/scatter density: 18 -> 9 (LOS for arrows).
        for _ in range(9):
            x = cx + int(self.rng.integers(-9, 10))
            y = cy + int(self.rng.integers(-9, 10))
            if 1 <= x < _SURFACE_SIZE - 1 and 1 <= y < _SURFACE_SIZE - 1:
                self._floors[0][y][x] = str(self.rng.choice([TILE_TREE, TILE_STONE, TILE_SAND]))
        _clear_area(self._floors[0], cx, cy, 2, _SURFACE_SIZE)
        self._floors[0][cy][cx + 2] = TILE_TABLE
        self._agent_x, self._agent_y = cx, cy
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "skeleton", cx, cy, floor=0, radius=6)
        self._inventory = {
            "bow": 1,
            "arrows": 6,
            "wood_sword": 1,
            "wood": 4,
            "stone": 4,
        }
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        self._bow_task_kills = 0
        self._bow_kills_this_step = 0
        return self._render_current_observation()

    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _record_bow_kill(self, mob: FullMob) -> None:
        if mob.get("type") == "skeleton" and mob.get("floor") == 0:
            self._bow_task_kills += 1
            self._bow_kills_this_step += 1

    def _attack_mob(self, mob: FullMob) -> float:
        was_target = mob.get("type") == "skeleton" and mob.get("floor") == 0
        reward = super()._attack_mob(mob)
        if was_target and mob not in self._mobs:
            self._record_bow_kill(mob)
        return reward

    def _attack_mob_kill(self, mob: FullMob) -> float:
        was_target = mob.get("type") == "skeleton" and mob.get("floor") == 0
        reward = super()._attack_mob_kill(mob)
        if was_target:
            self._record_bow_kill(mob)
        return reward

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._bow_kills_this_step = 0
        obs, reward, terminated, truncated, info = super()._step(action)
        kills = self._bow_kills_this_step
        # Replace parent's reward with kill credit (preserve per-step shaping).
        reward = kills / self._KILL_TARGET
        info["bow_task_kills"] = self._bow_task_kills
        if self._bow_task_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        # Death penalty applied additively so the kill that closed out the
        # episode still credits even if the killing blow also killed the
        # player (e.g. kamikaze final shot). subtask_success=False on death.
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        kills = getattr(self, "_bow_task_kills", 0)
        hud = f"{obs.hud}\nBow kills: {kills}/{self._KILL_TARGET}"
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=hud,
            message=obs.message,
        )


class CraftaxPotionTriageEnv(CraftaxFullEnv):
    """Use the hidden potion mapping under pressure, then win a fight."""

    tutorial_sections = (
        "overview",
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "legend:items", "legend:hud",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
        "combat:melee", "items:potions", "floors:0",
    )

    _KILL_TARGET = 2
    _BENEFICIAL_EFFECTS = frozenset({"heal_8"})
    _POTION_ACTION_TO_INDEX = {
        "DRINK_POTION_RED": 0,
        "DRINK_POTION_GREEN": 1,
        "DRINK_POTION_BLUE": 2,
        "DRINK_POTION_PINK": 3,
        "DRINK_POTION_CYAN": 4,
        "DRINK_POTION_YELLOW": 5,
    }
    _GATED_FLOORS = (0,)

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-potion-triage-v0"

    def _task_description(self) -> str:
        return (
            "You are at 2 HP with only three hidden-color potions: one heal, "
            "one poison, and one energy drain. Identify and drink the heal, "
            "then defeat 2 zombies. Reward: +1/2 for finding the heal potion "
            "and +1/4 per zombie kill, capped at two credited kills; -1 on "
            "death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        from glyphbench.envs.craftax.mechanics.potions import POTION_EFFECTS

        super()._reset(seed)
        self._current_floor = 0
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 5, _SURFACE_SIZE)
        for _ in range(12):
            x = cx + int(self.rng.integers(-7, 8))
            y = cy + int(self.rng.integers(-7, 8))
            if 1 <= x < _SURFACE_SIZE - 1 and 1 <= y < _SURFACE_SIZE - 1:
                self._floors[0][y][x] = str(self.rng.choice([TILE_TREE, TILE_STONE, TILE_WATER]))
        _clear_area(self._floors[0], cx, cy, 2, _SURFACE_SIZE)
        self._agent_x, self._agent_y = cx, cy
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "zombie", cx, cy, floor=0, radius=6)
        colors = ("red", "green", "blue", "pink", "cyan", "yellow")
        wanted_effects = {"heal_8", "poison_3", "energy_drain_3"}
        potions = {color: 0 for color in colors}
        for idx, effect_idx in enumerate(self._potion_mapping):
            if POTION_EFFECTS[effect_idx] in wanted_effects:
                potions[colors[idx]] = 1
        self._inventory = {"wood_sword": 1, "potions": potions}
        self._hp = 2
        self._mana = 0
        self._energy = 2
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._triage_found_benefit = False
        self._triage_kills = 0
        self._triage_kills_this_step = 0
        return self._render_current_observation()

    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _record_triage_kill(self, mob: FullMob) -> None:
        if mob.get("type") == "zombie" and mob.get("floor") == 0:
            self._triage_kills += 1
            self._triage_kills_this_step += 1

    def _attack_mob(self, mob: FullMob) -> float:
        was_target = mob.get("type") == "zombie" and mob.get("floor") == 0
        reward = super()._attack_mob(mob)
        if was_target and mob not in self._mobs:
            self._record_triage_kill(mob)
        return reward

    def _attack_mob_kill(self, mob: FullMob) -> float:
        was_target = mob.get("type") == "zombie" and mob.get("floor") == 0
        reward = super()._attack_mob_kill(mob)
        if was_target:
            self._record_triage_kill(mob)
        return reward

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        from glyphbench.envs.craftax.mechanics.potions import POTION_EFFECTS

        action_name = self.action_spec.names[action]
        potion_idx = self._POTION_ACTION_TO_INDEX.get(action_name)
        potion_effect = None
        if potion_idx is not None:
            color = action_name.removeprefix("DRINK_POTION_").lower()
            potions = self._inventory.get("potions", {})
            if potions.get(color, 0) > 0:
                potion_effect = POTION_EFFECTS[self._potion_mapping[potion_idx]]

        self._triage_kills_this_step = 0
        obs, reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if (
            not self._triage_found_benefit
            and potion_effect in self._BENEFICIAL_EFFECTS
        ):
            self._triage_found_benefit = True
            reward += 0.5
        kills = self._triage_kills_this_step
        if kills > 0:
            previous = max(0, self._triage_kills - kills)
            old_capped = min(previous, self._KILL_TARGET)
            new_capped = min(self._triage_kills, self._KILL_TARGET)
            reward += (new_capped - old_capped) * (0.5 / self._KILL_TARGET)
        info["triage_found_benefit"] = self._triage_found_benefit
        info["triage_kills"] = self._triage_kills
        if self._triage_found_benefit and self._triage_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        # Apply -1 death penalty additively (preserves benefit + kill credit).
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


# ===================================================================
# 15-18: Crafting Chain Tasks (surface scenario)
# ===================================================================

class CraftaxCraftIronSetEnv(CraftaxScenarioEnv):
    """Start near trees, stone, iron, table, furnace.
    Goal: craft iron pickaxe + iron sword. Max 480 steps."""

    # Manage death penalty here so milestone bonuses can't soften the
    # terminal -1.0 (e.g. "craft sword + die same tick" would otherwise
    # yield -0.5 instead of -1).
    _emit_death_penalty: bool = False

    # Day length is 200 steps; the 480-step budget reaches nightfall and
    # triggers night mob spawns. Disclose this in the prompt.
    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-craft-ironset-v0"

    def _task_description(self) -> str:
        return (
            "Resources (trees, stone, coal, iron) are nearby. Craft an iron pickaxe "
            "AND an iron sword. Chain: chop wood → place table → craft wood "
            "pickaxe → mine stone → place furnace → craft stone pickaxe → "
            "mine coal and iron → craft iron tools (each needs 1 wood + "
            "1 stone + 1 iron + 1 coal, and nearby table+furnace). "
            "Reward: +1/2 for iron pickaxe, +1/2 for iron sword."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []  # No mobs for crafting focus
        _cx, _cy = self._agent_x, self._agent_y
        _resource_field(
            self,
            {
                (-5, -4): TILE_TREE, (-5, -2): TILE_TREE,
                (-4, -5): TILE_TREE, (-3, -4): TILE_TREE,
                (-2, -5): TILE_TREE, (-6, -1): TILE_TREE,
                (4, -4): TILE_STONE, (5, -3): TILE_STONE,
                (4, -2): TILE_STONE, (6, -1): TILE_STONE,
                (5, 1): TILE_STONE, (4, 2): TILE_STONE,
                (-6, 3): TILE_COAL, (-4, 3): TILE_COAL,
                (-2, 3): TILE_COAL, (0, 4): TILE_COAL,
                (-3, 5): TILE_IRON, (-1, 5): TILE_IRON,
                (1, 6): TILE_IRON, (3, 5): TILE_IRON,
            },
        )
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._has_iron_pick = False
        self._has_iron_sword = False
        return self._render_current_observation()

    # Suppress parent achievement rewards; this focused subtask defines its
    # own two milestone rewards.
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Credit milestones FIRST (preserves training signal even on the
        # death tick — a "craft + die same step" still gets credit for
        # the craft, just with -1 applied below). Cumulative return stays
        # in [-1, +1] because milestones sum to at most +1.
        bonus = 0.0
        if self._inventory.get("iron_pickaxe", 0) > 0 and not self._has_iron_pick:
            self._has_iron_pick = True
            bonus += 0.5
        if self._inventory.get("iron_sword", 0) > 0 and not self._has_iron_sword:
            self._has_iron_sword = True
            bonus += 0.5
        reward += bonus
        # Apply death penalty additively (never as an override that drops
        # earned milestone progress).
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        elif self._has_iron_pick and self._has_iron_sword:
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxMineIronEnv(CraftaxScenarioEnv):
    """Stone pickaxe in inventory; iron deposits are scattered outside the
    spawn pocket. Goal: mine 5 iron ore within the 140-step default horizon.

    (Renamed from CraftaxSmeltIronEnv. The old env id was misleading —
    the Craftax implementation does not have a separate smelting step;
    DO on an iron tile with a stone pickaxe directly produces an
    ``iron`` inventory entry, treated as the smelted bar by the
    crafting recipes. The env id was changed to reflect what the env
    actually checks.)"""

    # Pattern A: +1/N per iron mined (sums to +1.0 at N iron).
    _IRON_TARGET = 5

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 140) -> None:
        super().__init__(max_turns=max_turns)
        self._iron_credited: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-mine-iron-v0"

    def _task_description(self) -> str:
        return (
            "Iron deposits (I) are scattered through a seed-generated quarry "
            "that keeps the agent out of the center pocket. You have a "
            "stone pickaxe. Mine 5 iron ore. Face iron tiles and use DO to "
            "mine. Reward: +1/5 per newly credited iron mined (sums to +1.0), "
            "-1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        _clear_area(self._world, cx, cy, 4, self._WORLD_SIZE)
        # Parent worldgen may leave natural iron just outside the cleared
        # square, making seed 0 a short local mining task. Remove nearby iron
        # first, then place the curated quarry farther out.
        for y, row in enumerate(self._world):
            for x, tile in enumerate(row):
                if tile == TILE_IRON and abs(x - cx) + abs(y - cy) < 9:
                    self._world[y][x] = TILE_GRASS
        # Keep same-seed resets deterministic but make the initial rendered
        # pocket visibly seed-dependent without placing target iron nearby.
        visible_sand = [
            (cx - 4, cy - 3),
            (cx - 3, cy - 3),
            (cx - 4, cy + 3),
            (cx + 4, cy - 3),
            (cx + 3, cy + 3),
            (cx - 2, cy + 3),
        ]
        sx, sy = visible_sand[seed % len(visible_sand)]
        if 0 <= sx < self._WORLD_SIZE and 0 <= sy < self._WORLD_SIZE:
            self._world[sy][sx] = TILE_SAND
        _scatter_scenario_tiles(
            self,
            cx,
            cy,
            count=14,
            radius=10,
            min_manhattan=6,
            tiles=(TILE_STONE, TILE_COAL, TILE_SAND),
        )
        for _ in range(8):
            for _attempt in range(60):
                dx = int(self.rng.integers(-13, 14))
                dy = int(self.rng.integers(-13, 14))
                if abs(dx) + abs(dy) < 9:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                    self._world[y][x] = TILE_IRON
                    break
        self._inventory = {"stone_pickaxe": 1}
        self._iron_credited = 0
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        new_iron = self._inventory.get("iron", 0)
        # Pattern A: +1/N per fresh iron, capped at the target. Use a
        # monotone credited counter rather than the current inventory delta,
        # so crafting/spending iron cannot make later mining earn duplicate
        # credit above the +1.0 episode cap.
        previous_credited = self._iron_credited
        self._iron_credited = max(
            self._iron_credited,
            min(new_iron, self._IRON_TARGET),
        )
        gained = max(
            0,
            self._iron_credited - previous_credited,
        )
        if gained > 0:
            reward += gained / self._IRON_TARGET
        if new_iron >= self._IRON_TARGET:
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxBuildShelterEnv(CraftaxScenarioEnv):
    """Mine stone, then place blocks to surround yourself."""

    _SHELTER_TARGET = 6

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)
        self._shelter_blocks_placed: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-build-shelter-v0"

    def _task_description(self) -> str:
        return (
            "Build a six-block stone shelter while zombies close in. Mine "
            "stone from the nearby quarry using your wood pickaxe, then place "
            "6 stone blocks on empty grass tiles with PLACE_STONE. Face the "
            "target tile before placing. Reward: +1/6 per placed block, up "
            "to +1; -1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        # Keep the placement cells clear, but preserve/seed nearby terrain
        # so different seeds render as different starts.
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        _scatter_scenario_tiles(
            self,
            cx,
            cy,
            count=18,
            radius=7,
            min_manhattan=3,
            tiles=(TILE_TREE, TILE_STONE, TILE_STONE, TILE_SAND),
        )
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        self._inventory = {"wood_pickaxe": 1}
        for _ in range(3):
            _place_scenario_mob(self, "zombie", cx, cy, radius=7)
        self._shelter_blocks_placed = 0
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_placed = sum(row.count(TILE_PLACED_STONE) for row in self._world)
        obs, reward, terminated, truncated, info = super()._step(action)
        new_placed = sum(row.count(TILE_PLACED_STONE) for row in self._world)
        placed_delta = max(0, new_placed - old_placed)
        remaining = max(0, self._SHELTER_TARGET - self._shelter_blocks_placed)
        credited = min(placed_delta, remaining)
        self._shelter_blocks_placed += credited

        reward = credited * (1.0 / self._SHELTER_TARGET)
        info["shelter_blocks_placed"] = self._shelter_blocks_placed
        if self._shelter_blocks_placed >= self._SHELTER_TARGET:
            terminated = True
            info["subtask_success"] = True
        # Apply -1 death penalty additively so the block-placement progress
        # earned this tick is preserved in the cumulative return.
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxPlantFarmEnv(CraftaxScenarioEnv):
    """Start with saplings. Plant, grow, harvest.
    Goal: eat 4 ripe plants. Max 200 steps."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._plants_harvested: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-plant-farm-v0"

    # Pattern A: 4 plants to harvest, +1/N per harvest (sum to +1.0).
    _HARVEST_TARGET = 4

    def _task_description(self) -> str:
        return (
            "You have 6 saplings and a wood sword. Harvest (eat) 4 ripe "
            "plants while zombies pressure the farm. Face grass and "
            "PLACE_PLANT to plant a sapling (;); wait ~20 steps for it to "
            "ripen into a ripe plant (*); face the ripe plant and EAT_PLANT "
            "to harvest. Reward: +1/4 per plant eaten (sum to +1.0 on full "
            "harvest); -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        self._plants_harvested = 0
        cx, cy = self._agent_x, self._agent_y
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        _scatter_scenario_tiles(
            self,
            cx,
            cy,
            count=14,
            radius=6,
            min_manhattan=3,
            tiles=(TILE_TREE, TILE_WATER, TILE_SAND),
        )
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        for _ in range(2):
            _place_scenario_mob(self, "zombie", cx, cy, radius=7)
        self._inventory = {"sapling": 6, "wood_sword": 1}
        self._hp = 9
        self._food = 5  # Not full, so eating is useful
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        action_name = self.action_spec.names[action]
        was_facing_ripe = False
        if action_name == "EAT_PLANT":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            was_facing_ripe = (
                0 <= fx < self._WORLD_SIZE
                and 0 <= fy < self._WORLD_SIZE
                and self._world[fy][fx] == TILE_RIPE_PLANT
            )
        old_harvested = self._plants_harvested
        obs, reward, terminated, truncated, info = super()._step(action)
        if was_facing_ripe:
            self._plants_harvested += 1
        # Replace parent reward with structural +1/N per harvest.
        new_harvests = self._plants_harvested - old_harvested
        reward = new_harvests * (1.0 / self._HARVEST_TARGET)
        if self._plants_harvested >= self._HARVEST_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


# ===================================================================
# 19-21: Exploration Tasks
# ===================================================================

class CraftaxFindWaterEnv(CraftaxScenarioEnv):
    """Start in a desert-like area. Navigate to find water. Max 180 steps."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-find-water-v0"

    def _task_description(self) -> str:
        return (
            "You are in a dry sandy area where water is scarce. Find a water "
            "tile and use DRINK_WATER facing it. Explore in all directions — "
            "water is somewhere in the world but not near your start, and a "
            "zombie is stalking the route. Reward: +1/4 when you get into "
            "the water's vicinity and +3/4 for drinking it (terminal), -1 "
            "on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        # Convert nearby water and grass to sand (desert)
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                ):
                    tile = self._world[y][x]
                    if tile in (TILE_WATER, TILE_GRASS):
                        self._world[y][x] = TILE_SAND
        # Ensure walkable start
        self._world[cy][cx] = TILE_SAND
        # Solvability fallback: if the wipe erased all water, place 1-3
        # water tiles deterministically at distance 18-22 from the player so
        # the env is always winnable. (B5: reverse-flow.)
        self._ensure_water_exists(cx, cy)
        _place_scenario_mob(self, "zombie", cx, cy, radius=8)
        self._inventory = {}
        self._hp = 9
        self._water = 1  # Thirst pressure without making the route impossible
        self._food = _MAX_FOOD
        self._energy = _MAX_ENERGY
        self._found_water = False
        self._water_midpoint_rewarded = False
        self._initial_water_distance = self._nearest_water_distance()
        return self._render_current_observation()

    def _ensure_water_exists(self, cx: int, cy: int) -> None:
        """If no water survived the wipe, place 1-3 water tiles at d=24..30."""
        has_water = any(
            tile == TILE_WATER
            for row in self._world
            for tile in row
        )
        if has_water:
            return
        # Place 1-3 water tiles in a deterministic ring around the player.
        # Use rng so the count and placement are seed-varied but reproducible.
        target_count = int(self.rng.integers(1, 4))  # 1, 2, or 3
        placed = 0
        for _attempt in range(200):
            if placed >= target_count:
                break
            # Sample distance 24..30 and a random angle.
            dist = int(self.rng.integers(24, 31))
            # Use cardinal+diagonal offsets that yield Chebyshev=dist.
            dx = int(self.rng.integers(-dist, dist + 1))
            dy_sign = 1 if int(self.rng.integers(0, 2)) == 0 else -1
            dy = dy_sign * (dist - abs(dx) if abs(dx) < dist else 0)
            if dx == 0 and dy == 0:
                continue
            x, y = cx + dx, cy + dy
            if not (
                0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE
            ):
                continue
            # Don't overwrite existing water (placed earlier in this loop).
            if self._world[y][x] == TILE_WATER:
                continue
            self._world[y][x] = TILE_WATER
            placed += 1

    def _nearest_water_distance(self) -> int:
        cx, cy = self._agent_x, self._agent_y
        best = self._WORLD_SIZE * 2
        for y, row in enumerate(self._world):
            for x, tile in enumerate(row):
                if tile == TILE_WATER:
                    best = min(best, abs(x - cx) + abs(y - cy))
        return best

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        action_name = self.action_spec.names[action]
        drank_from_water = False
        if action_name == "DRINK_WATER":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            drank_from_water = (
                0 <= fx < self._WORLD_SIZE
                and 0 <= fy < self._WORLD_SIZE
                and self._world[fy][fx] == TILE_WATER
            )
        old_water = self._water
        obs, reward, terminated, truncated, info = super()._step(action)
        # Credit midpoint + found-water progress first; death penalty is
        # applied additively below so progress earned on the death tick
        # remains in the cumulative return.
        reward = 0.0
        current_distance = self._nearest_water_distance()
        midpoint_threshold = max(2, self._initial_water_distance // 2)
        if not self._water_midpoint_rewarded and current_distance <= midpoint_threshold:
            self._water_midpoint_rewarded = True
            reward += 0.25
        if (drank_from_water or self._water > old_water) and not self._found_water:
            self._found_water = True
            reward += 0.75
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxFindDiamondEnv(CraftaxScenarioEnv):
    """Start with iron pickaxe. Find and mine a diamond. Max 220 steps.
    Pattern C: terminal +1 on diamond mined."""

    tutorial_sections = (
        "legend:player", "legend:terrain",
        "survival:hp_food_drink", "survival:energy_sleep",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 220) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-find-diamond-v0"

    def _task_description(self) -> str:
        return (
            "Diamonds are rare and scattered across the world. You have an "
            "iron pickaxe (required to mine diamond). Find a diamond tile "
            "and mine it with DO. Explore widely — diamonds may be far from "
            "your start, and a zombie can interrupt a careless straight-line "
            "search. Reward: +1 for mining a diamond (Pattern C, terminal), "
            "-1 on death."
        )

    # Seed-varied search distance for the guaranteed diamond.
    _DIAMOND_MIN_DIST = 18
    _DIAMOND_MAX_DIST = 34

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        self._inventory = {"iron_pickaxe": 1}
        self._hp = 9
        self._food = 6
        self._water = 6
        self._energy = _MAX_ENERGY
        # Retune: randomize the agent's surface start so the goal is not at a
        # fixed offset from a fixed origin. Pick a walkable cell near the map
        # center (worldgen guarantees a cleared center region).
        size = self._WORLD_SIZE
        c = size // 2
        for _attempt in range(60):
            sx = int(self.rng.integers(c - 6, c + 7))
            sy = int(self.rng.integers(c - 6, c + 7))
            if (
                0 <= sx < size
                and 0 <= sy < size
                and self._world[sy][sx] in SURFACE_WALKABLE
            ):
                self._agent_x, self._agent_y = sx, sy
                break
        # Place a single guaranteed-reachable diamond at a seed-randomized
        # direction and Manhattan/BFS distance (no fixed 10-east corridor); the
        # agent must navigate natural terrain to reach it. Natural worldgen may
        # still place additional diamonds farther away.
        self._diamond_pos = _place_reachable_diamond(
            self,
            (self._agent_x, self._agent_y),
            self._DIAMOND_MIN_DIST,
            self._DIAMOND_MAX_DIST,
        )
        _place_scenario_mob(self, "zombie", self._agent_x, self._agent_y, radius=10)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_diamond = self._inventory.get("diamond", 0)
        obs, reward, terminated, truncated, info = super()._step(action)
        # Pattern C: ignore parent reward, terminal +1 on diamond mined.
        reward = 0.0
        new_diamond = self._inventory.get("diamond", 0)
        if new_diamond > old_diamond:
            reward += 1.0
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxReachDungeonEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Start on surface. Find the dungeon entrance (stairs down).
    Max 180 steps."""

    tutorial_sections = (
        "overview",
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items", "legend:projectiles", "legend:hud",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:rest",
        "survival:day_night",
        "combat:melee", "combat:ranged_player", "combat:ranged_mob",
        "combat:armor", "combat:projectiles",
        "magic:spells", "magic:books", "magic:enchants",
        "crafting:wood", "crafting:stone", "crafting:iron", "crafting:diamond",
        "crafting:placement", "crafting:arrows",
        "items:resources", "items:bow", "items:potions",
        "progression:xp", "progression:attributes",
        "floors:0", "floors:1", "floors:navigation",
    )

    _GATED_FLOORS = (0,)
    _EXIT_KILL_REQUIREMENT = 2

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-reach-dungeon-v0"

    def _task_description(self) -> str:
        return (
            "A dungeon entrance is somewhere on the surface. Find the stairs "
            "down (⇣), defeat 2 hostile surface monsters to open the stair "
            "seal, then use DESCEND to enter the dungeon. The stairs are "
            "placed near the center of the map but you must navigate to them. "
            "Reward: +1 for entering the dungeon."
        )

    # Seed-varied spawn offset range from the entrance (Chebyshev radius).
    _SPAWN_MIN_OFFSET = 6
    _SPAWN_MAX_OFFSET = 11

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 0
        entrance = self._stairs_down_pos.get(0)
        if entrance:
            # Retune: randomize the spawn position relative to the (fixed)
            # entrance per seed — a random heading and distance — so the route
            # is no longer a fixed (+8,+8) diagonal that an open-loop "march SE"
            # script can memorize. The agent must read the viewport to find the
            # staircase.
            size = len(self._floors[0])
            ex, ey = entrance
            sx, sy = ex, ey
            for _attempt in range(80):
                ox = int(self.rng.integers(
                    -self._SPAWN_MAX_OFFSET, self._SPAWN_MAX_OFFSET + 1))
                oy = int(self.rng.integers(
                    -self._SPAWN_MAX_OFFSET, self._SPAWN_MAX_OFFSET + 1))
                if max(abs(ox), abs(oy)) < self._SPAWN_MIN_OFFSET:
                    continue
                cx = max(1, min(size - 2, ex + ox))
                cy = max(1, min(size - 2, ey + oy))
                if (cx, cy) != (ex, ey):
                    sx, sy = cx, cy
                    break
            # Spawn must stand on a walkable tile.
            surface = self._floors[0]
            if surface[sy][sx] not in SURFACE_WALKABLE:
                surface[sy][sx] = TILE_GRASS
            self._agent_x, self._agent_y = sx, sy
            # Retune: guarantee the staircase is reachable from spawn (the prior
            # code left ~18% of seeds with a blocked entrance).
            _carve_surface_route(self, (sx, sy), entrance)
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        _ensure_floor_gate_monsters_for(self, 0, self._EXIT_KILL_REQUIREMENT, cap=True)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 1, terminated, info
        )
        return obs, reward, terminated, truncated, info


# ===================================================================
# 22-23: Economy/Achievement Tasks
# ===================================================================

class CraftaxFirstDayEnv(CraftaxScenarioEnv):
    """Complete as many achievements as possible in 1 day cycle plus a
    short night slice (220 steps). Start from scratch. Reward: +1/22 per
    achievement unlocked."""

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )
    _ALL_ACHIEVEMENTS = (
        "collect_wood",
        "place_table",
        "make_wood_pickaxe",
        "collect_stone",
        "place_furnace",
        "make_stone_pickaxe",
        "collect_iron",
        "collect_coal",
        "place_stone",
        "collect_drink",
        "collect_sapling",
        "place_plant",
        "eat_plant",
        "defeat_zombie",
        "defeat_skeleton",
        "wake_up",
        "collect_diamond",
        "make_iron_pickaxe",
        "make_iron_sword",
        "make_wood_sword",
        "make_stone_sword",
        "eat_cow",
    )

    def __init__(self, max_turns: int = 220) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-firstday-v0"

    def _task_description(self) -> str:
        return (
            "You have 220 steps to accomplish as much as possible. Use the "
            "day to gather resources and craft tools, then use the short "
            "night slice to fight mobs. Start from scratch. Gather "
            "resources, craft tools, fight mobs. "
            "Reward: +1/22 per achievement unlocked (sums to +1.0 if all 22 "
            "Crafter achievements are reached); -1 on death. Achievements: "
            "collect_wood, place_table, make_wood_pickaxe, collect_stone, "
            "place_furnace, make_stone_pickaxe, collect_iron, collect_coal, "
            "place_stone, collect_drink, collect_sapling, place_plant, "
            "eat_plant, defeat_zombie, defeat_skeleton, wake_up, "
            "collect_diamond, make_iron_pickaxe, make_iron_sword, "
            "make_wood_sword, make_stone_sword, eat_cow."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Achievement reward is already handled by parent (+1 per unlock)
        info["total_achievements"] = len(self._achievements_unlocked)
        return obs, reward, terminated, truncated, info


class CraftaxSpeedrunEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Reach the boss as fast as possible. Start with endgame gear.
    Navigate through dungeon floors. Max 500 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "items:bow", "floors:0", "floors:1", "floors:2", "floors:3",
        "floors:4", "floors:5", "floors:navigation",
    )

    # Pattern A (milestone): 5 floor transitions to reach floor 5,
    # each worth +1/5 = +0.2 (sum to +1.0 on success).
    _FLOOR_TARGET = 5
    _PER_FLOOR_REWARD = 1.0 / 5
    _GATED_FLOORS = (0, 1, 2, 3, 4)
    _EXIT_KILL_REQUIREMENT = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-speedrun-v0"

    def _task_description(self) -> str:
        return (
            "Race to dungeon floor 5. You start on the surface (floor 0) "
            "right next to the stairs down, with endgame gear (diamond sword, "
            "full diamond armor, all spells learned, fire-enchanted weapon, "
            "bow, and arrows). Each floor's stairs are sealed until you defeat "
            "2 hostile monsters on that floor, then use DESCEND and repeat. "
            "Reward: "
            "+1/5 per new floor reached, episode ends on reaching floor 5. "
            "Time limit: 500 steps."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 0
        # Start near, but not on, the dungeon entrance.
        entrance = self._stairs_down_pos.get(0)
        if entrance:
            self._agent_x = max(5, entrance[0] - 3)
            self._agent_y = max(5, entrance[1] - 3)
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 20,
            "wood": 10,
            "coal": 10,
        }
        self._sword_enchantment = 1  # fire-enchanted by default (phase γ T10γ)
        # Phase γ T03γ: armour tracked in _armor_slots (4-slot dict).
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # all diamond tier
        _learn_all_spells(self)
        _give_potions(self, count=1)
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        self._deepest_floor = 0
        saved = (self._current_floor, self._agent_x, self._agent_y, self._facing)
        _ensure_floor_gate_monsters_for(self, 0, self._EXIT_KILL_REQUIREMENT, cap=True)
        for floor in (1, 2, 3, 4):
            _move_player_to_floor_start(self, floor)
            _ensure_floor_gate_monsters_for(
                self,
                floor,
                self._EXIT_KILL_REQUIREMENT,
                cap=True,
            )
        self._current_floor, self._agent_x, self._agent_y, self._facing = saved
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Pattern A: replace any parent-emitted reward with structural
        # per-floor milestones. Each new floor contributes +1/5 so the
        # sum on success is +1.0.
        reward = 0.0
        if self._current_floor > self._deepest_floor:
            floors_gained = self._current_floor - self._deepest_floor
            self._deepest_floor = self._current_floor
            reward = floors_gained * self._PER_FLOOR_REWARD
            if self._current_floor >= self._FLOOR_TARGET:
                terminated = True
                info["subtask_success"] = True
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        info["deepest_floor"] = self._deepest_floor
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Bootstrap / chain tasks — start from nothing and force the agent through
# the full Craftax mining + crafting dependency chain.
# ---------------------------------------------------------------------------


def _resource_field(env: CraftaxScenarioEnv, layout: dict[tuple[int, int], str]) -> None:
    """Helper for bootstrap-task ``_reset``: clear a small arena around
    the agent and stamp a seed-varied transform of specific resource offsets.
    ``layout`` maps (dx, dy) → tile char.
    """
    cx, cy = env._agent_x, env._agent_y
    for dx in range(-7, 8):
        for dy in range(-7, 8):
            x, y = cx + dx, cy + dy
            if 0 <= x < env._WORLD_SIZE and 0 <= y < env._WORLD_SIZE:
                env._world[y][x] = TILE_GRASS
    rotations = int(env.rng.integers(0, 4))
    mirror = bool(env.rng.integers(0, 2))

    def transform(dx: int, dy: int) -> tuple[int, int]:
        if mirror:
            dx = -dx
        for _ in range(rotations):
            dx, dy = -dy, dx
        return dx, dy

    for (dx, dy), tile in layout.items():
        dx, dy = transform(dx, dy)
        if dx == 0 and dy == 0:
            dx = 1
        x, y = cx + dx, cy + dy
        if 0 <= x < env._WORLD_SIZE and 0 <= y < env._WORLD_SIZE:
            env._world[y][x] = tile


class CraftaxIronBootstrapEnv(CraftaxScenarioEnv):
    """Empty inventory, blank meadow with ONE cluster of trees, stone,
    coal and iron ore. Goal: walk the full bootstrap chain — chop wood
    → place a crafting table → craft a wood pickaxe → mine stone →
    place a furnace → craft a stone pickaxe → mine 1 iron ore. Termination
    fires when both ``inventory["iron"] >= 1`` AND a furnace tile
    exists in the curated arena (so the agent really did place one,
    not just inherit it).
    """

    # Pattern A: 7 milestones (wood, table, wood_pickaxe, stone, furnace,
    # stone_pickaxe, iron_first) → +1/7 each (sums to +1.0 on success).
    _MILESTONE_COUNT = 7

    # Manage death penalty here so milestone bonuses can't soften the
    # terminal -1.0 (e.g. "mine iron + die same tick" would otherwise
    # yield -6/7 instead of -1).
    _emit_death_penalty: bool = False

    # Day length is 200 steps; this subtask's 320-step budget can reach
    # nightfall and trigger night mob spawns. Disclose this in the prompt.
    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 320) -> None:
        super().__init__(max_turns=max_turns)
        self._reached_iron: bool = False
        self._placed_furnace_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-iron-bootstrap-v0"

    def _task_description(self) -> str:
        return (
            "You start with NOTHING. The arena contains trees, stone, coal, "
            "and iron ore. Goal: mine 1 iron ore AND place 1 furnace. "
            "Required chain (no shortcuts): chop wood with DO → PLACE_TABLE "
            "(2 wood) → MAKE_WOOD_PICKAXE → mine stone → PLACE_FURNACE "
            "(1 stone) → MAKE_STONE_PICKAXE → mine iron with stone pickaxe. "
            "Reward: +1/7 per first-time milestone (sums to +1.0 on success)."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        # Set up curated arena: trees N+NE, stone S, coal SE, iron W.
        # Plenty of redundancy so the agent has slack.
        layout: dict[tuple[int, int], str] = {}
        for dy in (-3, -2, -1):
            layout[(0, dy)] = TILE_TREE
        for dy in (-3, -2):
            layout[(1, dy)] = TILE_TREE
        for dx in (1, 2, 3):
            layout[(dx, 1)] = TILE_STONE
            layout[(dx, 2)] = TILE_STONE
        for dx in (1, 2):
            layout[(dx, 3)] = TILE_COAL
        for dx in (-3, -2, -1):
            layout[(dx, 0)] = TILE_IRON
        _resource_field(self, layout)
        # Reset progress flags so reward shaping doesn't double-count
        # across episode resets.
        self._reached_iron = False
        self._placed_furnace_count = 0
        self._milestones: set[str] = set()
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _count_furnaces(self) -> int:
        n = 0
        for row in self._world:
            for ch in row:
                if ch == TILE_FURNACE:
                    n += 1
        return n

    def _milestone(self, key: str, points: float) -> float:
        if key in self._milestones:
            return 0.0
        self._milestones.add(key)
        return float(points)

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Always credit new milestones this step, including the death tick,
        # so cumulative return preserves the agent's training-signal of how
        # far it got before dying. Death penalty is applied additively
        # below — never as an override that drops earned progress.
        inv = self._inventory
        per = 1.0 / self._MILESTONE_COUNT
        if inv.get("wood", 0) >= 1:
            reward += self._milestone("wood", per)
        if inv.get("stone", 0) >= 1:
            reward += self._milestone("stone", per)
        if inv.get("wood_pickaxe", 0) >= 1:
            reward += self._milestone("wood_pickaxe", per)
        if inv.get("stone_pickaxe", 0) >= 1:
            reward += self._milestone("stone_pickaxe", per)
        # Detect placement events by counting tiles (cheap on a small
        # curated arena).
        if "table" not in self._milestones:
            for row in self._world:
                if TILE_TABLE in row:
                    reward += self._milestone("table", per)
                    break
        if "furnace" not in self._milestones and self._count_furnaces() >= 1:
            reward += self._milestone("furnace", per)
        if inv.get("iron", 0) >= 1:
            reward += self._milestone("iron_first", per)
            self._reached_iron = True

        # Death precedence: -1 added on top of milestones credited this
        # tick. Cumulative return stays bounded in [-1, +1] because
        # sum(milestones) ≤ 1. subtask_success is False if dead.
        if self._hp <= 0:
            reward += -1.0
            terminated = True
            info["subtask_success"] = False
        elif (
            self._reached_iron
            and self._count_furnaces() >= 1
            and not terminated
        ):
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxDiamondBootstrapEnv(CraftaxIronBootstrapEnv):
    """Same arena as the iron bootstrap plus a diamond cluster, longer
    walltime, and the goal extended one more rung up the tool tree:
    forge an iron pickaxe and mine a diamond. Inherits all reward
    shaping from the iron bootstrap; adds two more milestones for
    iron-pickaxe craft and the first diamond. Termination fires on
    ``inventory["diamond"] >= 1``.
    """

    # Pattern A: 9 milestones total — 7 inherited (wood/table/wood-pickaxe/
    # stone/furnace/stone-pickaxe/iron_first) + iron_pickaxe + diamond_first.
    # Each contributes +1/9 (sums to +1.0 on diamond collection).
    _MILESTONE_COUNT = 9

    # Day length is 200 steps; the 420-step budget reaches nightfall and
    # triggers night mob spawns. Disclose this in the prompt.
    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 420) -> None:
        super().__init__(max_turns=max_turns)
        self._reached_diamond: bool = False

    def env_id(self) -> str:
        return "glyphbench/craftax-diamond-bootstrap-v0"

    def _task_description(self) -> str:
        return (
            "Like iron-bootstrap, but go one rung further: forge an iron "
            "pickaxe and mine a diamond. The arena contains trees, stone, "
            "coal, iron ore, AND a small diamond cluster. Required chain: "
            "wood → table → wood pickaxe → stone → furnace → stone pickaxe → "
            "iron ore + coal → iron pickaxe (1 wood + 1 stone + 1 iron + "
            "1 coal, table+furnace) → DO "
            "on diamond. Reward: +1/9 per first-time milestone (9 total: "
            "wood, table, wood-pickaxe, stone, furnace, stone-pickaxe, iron, "
            "iron-pickaxe, diamond) — sums to +1.0 on diamond collection."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Inherit iron-bootstrap arena, then add a diamond cluster.
        cx, cy = self._agent_x, self._agent_y
        for dx in (-3, -2):
            x, y = cx + dx, cy - 3
            if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                self._world[y][x] = TILE_DIAMOND
        self._reached_diamond = False
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Parent (CraftaxIronBootstrapEnv._step) already credited the
        # iron-tier milestones and, if the agent died, added -1 + reset
        # subtask_success=False. We need to also credit the diamond-tier
        # milestones this step (preserve training signal even on death)
        # and, if alive + iron-bootstrap goal was met but diamond not yet
        # collected, suppress the iron terminal.
        is_dead = self._hp <= 0
        if (
            terminated
            and not is_dead
            and not self._reached_diamond
            and self._reached_iron
            and self._count_furnaces() >= 1
        ):
            terminated = False
            info.pop("subtask_success", None)
        per = 1.0 / self._MILESTONE_COUNT
        if self._inventory.get("iron_pickaxe", 0) >= 1:
            reward += self._milestone("iron_pickaxe", per)
        if self._inventory.get("diamond", 0) >= 1:
            reward += self._milestone("diamond_first", per)
            if not self._reached_diamond and not terminated and not is_dead:
                terminated = True
                info["subtask_success"] = True
            self._reached_diamond = True
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Wave Defense — survive scripted zombie waves with limited stone for
# walls and a stone sword for fighting.
# ---------------------------------------------------------------------------


class CraftaxWaveDefenseEnv(CraftaxScenarioEnv):
    """Three scripted zombie waves over a 180-step episode. Agent
    starts with a stone sword and 6 stone for placing walls. Waves
    spawn at steps 1, 60, 120; each wave drops a small ring of zombies
    in the agent's vicinity. Success: alive at the end of step 180.

    Pattern A: 10 zombies total (3+3+4) → +0.1 per kill (sums to +1.0
    if every zombie is defeated). No step-survival bonus; no terminal
    bonus.
    """

    _WAVE_STEPS = (1, 60, 120)
    _WAVE_SIZES = (3, 3, 4)
    _KILL_TARGET = 10

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)
        self._waves_spawned: int = 0
        self._wave_kills: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-wave-defense-v0"

    def _task_description(self) -> str:
        return (
            "Three zombie waves spawn near you at steps 1, 60, 120 (sizes 3, "
            "3, 4). You have a stone sword (DO to attack adjacent zombies) "
            "and 6 stone (PLACE_STONE to wall off approaches). 9 HP, no "
            "armor. Waves appear outside immediate trap range and do not move "
            "until the next turn. Goal: kill all 10 zombies and survive to "
            "step 180. This is a scripted-wave combat drill, not upstream's "
            "stochastic night spawning: day/night and survival drains are "
            "disabled. "
            "Reward: +0.1 per kill (sums to +1.0 on a full clear)."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        _scatter_scenario_tiles(
            self,
            cx,
            cy,
            count=10,
            radius=6,
            min_manhattan=3,
            tiles=(TILE_TREE, TILE_STONE, TILE_SAND),
        )
        _clear_area(self._world, cx, cy, 2, self._WORLD_SIZE)
        self._inventory = {"stone_sword": 1, "stone": 6}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        # Disable day/night drains so the episode is purely combat.
        self._disable_survival_drain = True  # type: ignore[attr-defined]
        self._disable_day_night = True  # type: ignore[attr-defined]
        self._waves_spawned = 0
        self._wave_kills = 0
        return self._render_current_observation()

    def _spawn_wave(self, n_zombies: int) -> None:
        cx, cy = self._agent_x, self._agent_y
        # Try Manhattan-ring positions in a deterministic order so a
        # crowded arena still yields valid spawns.
        offsets = []
        for r in (6, 7, 8, 9):
            for dx in range(-r, r + 1):
                dy = r - abs(dx)
                offsets.append((dx, dy))
                if dy != 0:
                    offsets.append((dx, -dy))
        placed = 0
        for (dx, dy) in offsets:
            if placed >= n_zombies:
                break
            x, y = cx + dx, cy + dy
            if not (0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE):
                continue
            if self._world[y][x] not in SURFACE_WALKABLE:
                continue
            if (x, y) == (cx, cy) or self._mob_at(x, y):
                continue
            mob: Mob = {
                "type": "zombie",
                "x": x,
                "y": y,
                "hp": _MOB_STATS["zombie"]["hp"],
                "max_hp": _MOB_STATS["zombie"]["hp"],
                "attack_cooldown": 0,
            }
            self._mobs.append(mob)
            placed += 1

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock_achievement(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        prev_alive = len([m for m in self._mobs if m["type"] != "cow"])
        obs, reward, terminated, truncated, info = super()._step(action)
        # Pattern A: ignore parent reward; +1/N per kill, no step bonus,
        # no terminal bonus.
        reward = 0.0
        cur_alive = len([m for m in self._mobs if m["type"] != "cow"])
        if cur_alive < prev_alive:
            kills = prev_alive - cur_alive
            old_capped = min(self._wave_kills, self._KILL_TARGET)
            self._wave_kills += kills
            new_capped = min(self._wave_kills, self._KILL_TARGET)
            reward += (new_capped - old_capped) / self._KILL_TARGET
        # Spawn pending waves after the parent step. The player sees the
        # new wave in this returned observation, but those mobs do not get
        # an immediate same-turn AI move.
        for idx, step in enumerate(self._WAVE_STEPS):
            if (
                idx >= self._waves_spawned
                and self._turn >= step
                and not terminated
            ):
                self._spawn_wave(self._WAVE_SIZES[idx])
                self._waves_spawned = idx + 1
                obs = self._render_current_observation()
        episode_exhausted = truncated or self._turn >= self.max_turns
        if (
            episode_exhausted
            and self._hp > 0
            and self._waves_spawned == len(self._WAVE_STEPS)
            and self._wave_kills >= self._KILL_TARGET
        ):
            info["subtask_success"] = True
        info["wave_kills"] = self._wave_kills
        info["waves_spawned"] = self._waves_spawned
        return obs, reward, terminated, truncated, info



# ===================================================================
# Late-floor progression — descend stairs from each named dungeon floor.
# Floor 8 is covered by CraftaxNecromancerEnv (boss fight) below.
# ===================================================================


class CraftaxVaultsEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Vaults. Start with diamond sword + iron armor + spells.
    Goal: find stairs down. Combat is optional. Max 480 steps."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "survival:mana", "combat:elemental", "magic:spells",
        "magic:enchants", "items:bow", "items:potions", "items:gems",
        "floors:4", "floors:5", "floors:navigation",
    )

    _GATED_FLOORS = (4,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-vaults-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Vaults (floor 4, home of the fire enchant "
            "table Ⓔ). Find the stairs down (⇣) and use DESCEND to reach "
            f"the Troll Mines. {_FLOOR_GATE_TASK_TEXT}"
            "You start with a diamond sword, iron "
            "armor, learned spells, and potions. Use spells or enchanted "
            "attacks only if durable threats block the route. Reward: +1 "
            "for descending, -1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 4
        up_pos = self._stairs_up_pos.get(4)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "wood": 5,
            "coal": 5,
            "ruby": 1,
        }
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1  # iron tier
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_floor_nav_hp(self, 28)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        _ensure_floor_gate_monsters(self, 4)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 5, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxTrollMinesEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Troll Mines. Start with diamond gear + bow.
    Goal: find stairs down. Combat is optional. Max 240 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "items:bow", "items:gems", "floors:5", "floors:6",
        "floors:navigation",
    )

    _GATED_FLOORS = (5,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-troll-mines-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Troll Mines (floor 5). Find "
            "the stairs down (⇣) and use DESCEND to reach the Fire Realm. "
            f"{_FLOOR_GATE_TASK_TEXT}"
            "You start with a diamond sword, full diamond armor, bow + "
            "10 arrows, learned spells, and potions. Trolls hit hard; deep "
            "things kite from range, so avoid unnecessary fights. Reward: +1 "
            "for descending, -1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 5
        up_pos = self._stairs_up_pos.get(5)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 10,
            "wood": 5,
            "coal": 5,
        }
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # diamond tier
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_floor_nav_hp(self, 28)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        _ensure_floor_gate_monsters(self, 5)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 6, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxFireRealmEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Fire Realm — lava, fire-immune mobs. Start with
    ice-enchanted gear + iceball spell. Goal: find stairs down. Max 260 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "items:bow", "items:gems", "floors:6", "floors:7",
        "floors:navigation",
    )

    _GATED_FLOORS = (6,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fire-realm-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Fire Realm (floor 6 — lava + fire-immune "
            "mobs). Find the stairs down (⇣) and use DESCEND to reach "
            f"the Ice Realm. {_FLOOR_GATE_TASK_TEXT}"
            "Pigmen (p) and fire elementals (F) are immune to fire "
            "(0.9 physical defense, 1.0 fire defense) — use CAST_ICEBALL or "
            "your ice-enchanted diamond sword. You start with full diamond "
            "armor fire-enchanted (resists the realm's fire damage), an "
            "ice-enchanted sword, bow + 10 arrows, iceball spell learned, "
            "and potions. Avoid "
            "stepping on lava (♨, 2 HP/tick). Reward: +1 for descending, "
            "-1 on death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 6
        up_pos = self._stairs_up_pos.get(6)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 10,
            "wood": 5,
            "coal": 5,
        }
        self._sword_enchantment = 2  # ice-enchanted (offense vs fire-immune mobs)
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # diamond tier
        # Retune: the Fire Realm mobs deal FIRE-typed damage (pigman melee /
        # fire_elemental fireball), so survival requires FIRE resistance on the
        # armour, not ice. Fire-enchant all four slots: this cuts pigman melee
        # and elemental fireball from 7 to 3 dmg/hit, turning a literal sub-1-kill
        # death into a survivable kite-and-iceball fight. (The sword stays
        # ice-enchanted for offense, since the mobs are fire-immune.)
        _set_armor_enchants(self, {
            "helmet": 1, "chest": 1, "legs": 1, "boots": 1,
        })
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_floor_nav_hp(self, 28)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        _ensure_floor_gate_monsters(self, 6)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 7, terminated, info
        )
        return obs, reward, terminated, truncated, info


class CraftaxIceRealmEnv(_FloorExitGateMixin, CraftaxFullEnv):
    """Ice Realm — water, ice-immune mobs. Start with
    fire-enchanted gear + fireball spell. Goal: find stairs down. Max 260 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "items:bow", "items:gems", "floors:7", "floors:8",
        "floors:navigation",
    )

    _GATED_FLOORS = (7,)

    def __init__(self, max_turns: int = 480) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-ice-realm-v0"

    def _task_description(self) -> str:
        return (
            "You are in the Ice Realm (floor 7 — last floor before the "
            "boss). Find the stairs down (⇣) and use DESCEND to reach "
            f"the Graveyard. {_FLOOR_GATE_TASK_TEXT}"
            "Frost trolls (r) and ice elementals "
            "(i) are immune to ice (0.9 physical defense, 1.0 ice defense) "
            "— use CAST_FIREBALL or your fire-enchanted diamond sword. You "
            "start with full diamond armor ice-enchanted (resists the realm's "
            "ice damage), a fire-enchanted sword, bow + 10 arrows, fireball "
            "spell learned, and potions. Reward: +1 for descending, -1 on "
            "death, 0 on timeout."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 7
        up_pos = self._stairs_up_pos.get(7)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 10,
            "wood": 5,
            "coal": 5,
        }
        self._sword_enchantment = 1  # fire-enchanted (offense vs ice-immune mobs)
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # diamond tier
        # Retune: the Ice Realm mobs deal ICE-typed damage (frost_troll melee /
        # ice_elemental iceball), so survival requires ICE resistance on the
        # armour, not fire. Ice-enchant all four slots: this cuts frost_troll
        # melee and elemental iceball from 7 to 3 dmg/hit. (The sword stays
        # fire-enchanted for offense, since the mobs are ice-immune.)
        _set_armor_enchants(self, {
            "helmet": 2, "chest": 2, "legs": 2, "boots": 2,
        })
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_floor_nav_hp(self, 28)
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        _ensure_floor_gate_monsters(self, 7)
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        reward, terminated = _floor_transition_reward(
            self, old_floor, 8, terminated, info
        )
        return obs, reward, terminated, truncated, info


# ===================================================================
# Necromancer boss fight (floor 8) — the final boss in isolation.
# CraftaxFullEnv._step already wires the win condition (boss_progress
# >= 8 -> defeat_necromancer achievement).
# This env just spawns the agent on floor 8 with full kit so the
# fight is reachable without playing the full game.
# ===================================================================


class CraftaxNecromancerEnv(CraftaxFullEnv):
    """Final boss fight on floor 8. Start fully equipped on the Graveyard.
    Goal: land 8 hits on the vulnerable necromancer (n). Max 420 steps."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "legend:mobs:boss", "items:bow", "floors:8", "floors:navigation",
        "boss",
    )

    # Pattern D: +1.0 on necromancer kill, -1.0 on death.
    _DEATH_PENALTY = -1.0

    def __init__(self, max_turns: int = 420) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-necromancer-v0"

    def _task_description(self) -> str:
        return (
            "Defeat the Necromancer on floor 8 (Graveyard). The boss tile "
            "alternates between N (invulnerable, ignore) and n (vulnerable). "
            "Stand adjacent to n facing it, then use DO to land a hit "
            "(boss_progress += 1). Each hit triggers a 7-turn summon wave "
            "(zombies + skeletons spawn near you); kill them all, then wait "
            "for the timer to expire — the boss becomes vulnerable again. "
            "Repeat 8 times to win. You start with diamond sword + full "
            "diamond armor + fire- and ice-enchanted gear + both spells "
            "learned + bow + 20 arrows + potions. Floor 8 multiplies "
            "incoming damage by ×1.5 and freezes vitals (food/drink/energy "
            "do not decay). Reward: +1/8 per vulnerable hit, -1 on death; "
            "episode ends. Time limit: 420 steps."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 8
        # Floor 8 intentionally has no exit in upstream Craftax. Spawn near
        # a corner away from the centre boss tile.
        self._agent_x = 3
        self._agent_y = 3
        # Full endgame kit.
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 20,
            "wood": 10,
            "coal": 10,
            "ruby": 2,
            "sapphire": 2,
        }
        # Sword + bow fire-enchanted (deals fire damage on each swing/arrow);
        # armour mixed fire/ice for resistance to the summoned wave mobs.
        self._sword_enchantment = 1  # fire
        self._bow_enchantment = 1  # fire
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 2  # diamond tier
        _set_armor_enchants(self, {
            "helmet": 1, "chest": 2, "legs": 1, "boots": 2,
        })
        _learn_all_spells(self)
        _give_potions(self, count=1)
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        # Reset boss state so the fight starts fresh.
        self._boss_progress = 0
        self._boss_summon_timer = 0
        self._update_necromancer_tile_glyph()
        return self._render_current_observation()

    # Suppress parent achievement rewards (the subtask defines its own goal).
    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
        return 0.0

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_progress = self._boss_progress
        was_won = (
            self._boss_progress >= 8
            and "defeat_necromancer" in self._achievements_unlocked
        )
        obs, _parent_reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        gained = max(0, min(self._boss_progress, 8) - min(old_progress, 8))
        if gained > 0:
            reward += gained / 8.0
        if self._hp <= 0:
            reward = self._DEATH_PENALTY
            terminated = True
            return obs, reward, terminated, truncated, info
        won_now = (
            "defeat_necromancer" in self._achievements_unlocked and not was_won
        )
        if won_now:
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# P3 hard-task family — fork-transfer drills under the <=512-step cap.
# ===================================================================


class _FocusedRewardMixin:
    """Suppress achievement reward noise and expose coarse milestone bins."""

    _emit_death_penalty: bool = False
    _DEATH_PENALTY: float = -1.0

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)  # type: ignore[misc]
        self._stage_rewards: set[str] = set()
        return obs

    def _try_unlock(self, name: str) -> float:  # type: ignore[override]
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
        return 0.0

    def _stage_reward(self, key: str, value: float) -> float:
        if key in self._stage_rewards:
            return 0.0
        self._stage_rewards.add(key)
        return value


class _TargetKillMixin:
    """Track kills of task target mobs from melee, arrows, and spells."""

    _TARGET_TYPES: tuple[str, ...] = ()
    _TARGET_FLOOR: int | None = None

    def _reset_target_kills(self) -> None:
        self._target_kills = 0
        self._target_kills_this_step = 0

    def _is_target_mob(self, mob: FullMob) -> bool:
        if self._TARGET_TYPES and mob.get("type") not in self._TARGET_TYPES:
            return False
        return not (
            self._TARGET_FLOOR is not None and int(mob.get("floor", -1)) != self._TARGET_FLOOR
        )

    def _record_target_kill(self, mob: FullMob) -> None:
        if self._is_target_mob(mob):
            self._target_kills += 1
            self._target_kills_this_step += 1

    def _attack_mob(self, mob: FullMob) -> float:
        was_target = self._is_target_mob(mob)
        reward = super()._attack_mob(mob)  # type: ignore[misc]
        if was_target and mob not in self._mobs:
            self._record_target_kill(mob)
        return reward

    def _attack_mob_kill(self, mob: FullMob) -> float:
        was_target = self._is_target_mob(mob)
        reward = super()._attack_mob_kill(mob)  # type: ignore[misc]
        if was_target:
            self._record_target_kill(mob)
        return reward


def _move_player_to_floor_start(env: CraftaxFullEnv, floor: int) -> None:
    env._current_floor = floor
    pos = env._stairs_up_pos.get(floor)
    if pos is None:
        pos = (_DUNGEON_SIZE // 2, _DUNGEON_SIZE // 2)
    env._agent_x, env._agent_y = pos
    env._facing = (1, 0)


def _set_vitals(
    env: CraftaxFullEnv,
    *,
    hp: int = 9,
    food: int = _MAX_FOOD,
    water: int = _MAX_WATER,
    energy: int = _MAX_ENERGY,
    mana: int = _MAX_MANA,
) -> None:
    env._max_hp = hp
    env._hp = hp
    env._food = food
    env._water = water
    env._energy = energy
    env._mana = mana


def _prepare_surface_arena(
    env: CraftaxFullEnv,
    *,
    radius: int = 6,
    scatter: int = 12,
) -> tuple[int, int]:
    env._current_floor = 0
    cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
    _clear_area(env._floors[0], cx, cy, radius, _SURFACE_SIZE)
    for _ in range(scatter):
        x = cx + int(env.rng.integers(-(radius + 3), radius + 4))
        y = cy + int(env.rng.integers(-(radius + 3), radius + 4))
        if 1 <= x < _SURFACE_SIZE - 1 and 1 <= y < _SURFACE_SIZE - 1:
            env._floors[0][y][x] = str(env.rng.choice([TILE_TREE, TILE_STONE, TILE_SAND]))
    _clear_area(env._floors[0], cx, cy, 2, _SURFACE_SIZE)
    env._agent_x, env._agent_y = cx, cy
    return cx, cy


def _place_table_and_furnace_near(env: CraftaxFullEnv, x: int, y: int) -> None:
    grid = env._floors[env._current_floor]
    if 1 <= x + 1 < len(grid) - 1:
        grid[y][x + 1] = TILE_TABLE
    if 1 <= x < len(grid) - 1 and 1 <= y + 1 < len(grid) - 1:
        grid[y + 1][x] = TILE_FURNACE


def _spawn_surface_pressure_mob(
    env: CraftaxFullEnv,
    mob_type: str,
    *,
    min_dist: int = 5,
    max_dist: int = 8,
) -> bool:
    stats = FULL_MOB_STATS[mob_type]
    for _attempt in range(80):
        dx = int(env.rng.integers(-max_dist, max_dist + 1))
        dy = int(env.rng.integers(-max_dist, max_dist + 1))
        dist = abs(dx) + abs(dy)
        if dist < min_dist or dist > max_dist:
            continue
        x, y = env._agent_x + dx, env._agent_y + dy
        if (
            1 <= x < _SURFACE_SIZE - 1
            and 1 <= y < _SURFACE_SIZE - 1
            and env._floors[0][y][x] in SURFACE_WALKABLE
            and not env._mob_at(x, y, 0)
        ):
            env._mobs.append({
                "type": mob_type,
                "x": x,
                "y": y,
                "hp": stats["hp"],
                "max_hp": stats["hp"],
                "is_boss": False,
                "floor": 0,
                "attack_cooldown": 0,
            })
            return True
    return False


class _ContinuousNightPressureMixin:
    _NIGHT_SPAWN_INTERVAL = 15
    _NIGHT_SPAWN_LIMIT = 10
    _NIGHT_SPAWN_TYPES = ("zombie", "skeleton")

    def _advance_day_counter(self, steps: int = 1) -> None:
        for _ in range(steps):
            super()._advance_day_counter(1)  # type: ignore[misc]
            if self._current_floor != 0 or self._day_night != "night":
                continue
            if self._day_counter % self._NIGHT_SPAWN_INTERVAL != 0:
                continue
            hostiles = [
                m for m in self._mobs
                if m.get("floor") == 0 and m.get("type") != "cow"
            ]
            if len(hostiles) >= self._NIGHT_SPAWN_LIMIT:
                continue
            mtype = self._NIGHT_SPAWN_TYPES[
                int(self.rng.integers(0, len(self._NIGHT_SPAWN_TYPES)))
            ]
            _spawn_surface_pressure_mob(self, mtype)


class CraftaxTorchDescendEnv(
    _TargetKillMixin, _FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv
):
    """Dark-floor torch drill with a local 8-kill gate."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "crafting:torches", "floors:2", "floors:navigation",
    )
    _dungeon_ambient_light = 0.0
    _GATED_FLOORS = (2,)
    _EXIT_KILL_REQUIREMENT = 8
    _TARGET_FLOOR = 2
    _TARGET_TYPES = ("gnome_warrior", "gnome_archer")

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-torchdescend-v0"

    def _task_description(self) -> str:
        return (
            "Dark Gnomish Mines drill. Craft or place torches to extend the "
            "lit area, defeat 8 hostile floor mobs to open the stair seal, "
            "then stand on ⇣ and use DESCEND. Coarse rewards: +0.2 first "
            "torch craft, +0.2 first torch placement, +0.2 at 4 kills, "
            "+0.2 at 8 kills, +0.2 on DESCEND; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 2)
        grid = self._floors[2]
        _clear_dungeon_area(grid, self._agent_x, self._agent_y, 3, _DUNGEON_SIZE)
        _place_table_and_furnace_near(self, self._agent_x, self._agent_y)
        self._inventory = {
            "iron_sword": 1,
            "stone_pickaxe": 1,
            "wood": 3,
            "coal": 3,
            "torches": 1,
        }
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        _set_vitals(self, hp=12)
        _give_potions(self, count=1)
        _ensure_floor_gate_monsters_for(self, 2, self._EXIT_KILL_REQUIREMENT)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        kills = self._floor_task_kills.get(2, 0)
        if "make_torch" in self._achievements_unlocked or self._inventory.get("torches", 0) >= 4:
            reward += self._stage_reward("crafted_torch", 0.2)
        if self._torches_by_floor.get(2):
            reward += self._stage_reward("placed_torch", 0.2)
        if kills >= 4:
            reward += self._stage_reward("four_kills", 0.2)
        if kills >= 8:
            reward += self._stage_reward("eight_kills", 0.2)
        if old_floor == 2 and self._current_floor == 3:
            reward += self._stage_reward("descended", 0.2)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxSurviveNightHardEnv(
    _ContinuousNightPressureMixin, _FocusedRewardMixin, CraftaxFullEnv
):
    """Survive a continuous-spawn night with shelter resources."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS + (
        "survival:day_night", "crafting:torches",
    )
    _NIGHT_SPAWN_INTERVAL = 12
    _NIGHT_SPAWN_LIMIT = 9

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survivenight-v0"

    def _task_description(self) -> str:
        return (
            "Start just before sunset and survive until dawn under repeated "
            "night spawns. You have a stone sword, limited stone for shelter, "
            "and two torches. Coarse rewards: +0.25 for surviving 25, 50, "
            "and 75 night steps, +0.25 at dawn; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=6, scatter=14)
        self._mobs = []
        self._inventory = {"stone_sword": 1, "stone": 6, "torches": 2}
        _set_vitals(self, hp=9)
        self._day_counter = _DAY_LENGTH - 5
        self._day_night = "day"
        for _ in range(2):
            _spawn_surface_pressure_mob(self, "zombie", min_dist=6, max_dist=8)
        self._agent_x, self._agent_y = cx, cy
        return self._render_current_observation()

    def _night_elapsed(self) -> int:
        return min(_NIGHT_LENGTH, max(0, self._day_counter - _DAY_LENGTH))

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_phase = self._day_night
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        elapsed = self._night_elapsed()
        for threshold in (25, 50, 75):
            if elapsed >= threshold:
                reward += self._stage_reward(f"survive_{threshold}", 0.25)
        if old_phase == "night" and self._day_night == "day" and self._hp > 0:
            reward += self._stage_reward("dawn", 0.25)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxKiteSkeletonsEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Weak-weapon ranged-mob kiting drill."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS
    _TARGET_TYPES = ("skeleton",)
    _TARGET_FLOOR = 0
    _KILL_TARGET = 4
    _GATED_FLOORS = (0,)

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-kiteskeletons-v0"

    def _task_description(self) -> str:
        return (
            "Defeat 4 skeleton archers with only a wood sword. Use trees and "
            "stone cover to close distance and avoid arrows; rushing several "
            "archers at once is lethal. Rewards: +0.25 for the first kill, "
            "+0.25 at 2 kills, +0.5 at 4 kills; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=7, scatter=22)
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "skeleton", cx, cy, floor=0, radius=8)
        self._inventory = {"wood_sword": 1}
        _set_vitals(self, hp=9)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._target_kills >= 1:
            reward += self._stage_reward("one_kill", 0.25)
        if self._target_kills >= 2:
            reward += self._stage_reward("two_kills", 0.25)
        if self._target_kills >= self._KILL_TARGET:
            reward += self._stage_reward("clear", 0.5)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["target_kills"] = self._target_kills
        return obs, reward, terminated, truncated, info


class CraftaxDarkAmbushEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Fight hidden ranged mobs on a dark dungeon floor."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "crafting:torches", "floors:3", "floors:navigation",
    )
    _dungeon_ambient_light = 0.0
    _TARGET_TYPES = ("kobold",)
    _TARGET_FLOOR = 3
    _KILL_TARGET = 4
    _GATED_FLOORS = (3,)

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-dark-ambush-v0"

    def _task_description(self) -> str:
        return (
            "You are ambushed in a dark sewer. Mobs and projectiles in unlit "
            "cells are hidden; place torches to reveal kobolds before "
            "fighting. Rewards: +0.25 for placing a torch, +0.25 at 2 kills, "
            "+0.5 at 4 kills; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 3)
        grid = self._floors[3]
        _clear_dungeon_area(grid, self._agent_x, self._agent_y, 3, _DUNGEON_SIZE)
        self._mobs = [m for m in self._mobs if m.get("floor") != 3]
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "kobold", self._agent_x, self._agent_y, floor=3, radius=9)
        self._inventory = {"iron_sword": 1, "bow": 1, "arrows": 10, "torches": 4}
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        _set_vitals(self, hp=12)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._torches_by_floor.get(3):
            reward += self._stage_reward("torch", 0.25)
        if self._target_kills >= 2:
            reward += self._stage_reward("two_kills", 0.25)
        if self._target_kills >= self._KILL_TARGET:
            reward += self._stage_reward("clear", 0.5)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["target_kills"] = self._target_kills
        return obs, reward, terminated, truncated, info


class CraftaxFloorClear9HpEnv(_FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Single floor, 9 HP, uncapped local 8-kill stair gate."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "items:potions", "floors:1", "floors:navigation",
    )
    _GATED_FLOORS = (1,)
    _EXIT_KILL_REQUIREMENT = 8

    def __init__(self, max_turns: int = 400) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-floor-clear-9hp-v0"

    def _task_description(self) -> str:
        return (
            "Clear a Dungeon floor at the real 9-HP start. Defeat 8 hostile "
            "mobs to open the stairs, then DESCEND. Coarse rewards: +0.25 at "
            "4 kills, +0.5 at 8 kills, +0.25 on DESCEND; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 1)
        self._inventory = {"iron_sword": 1, "stone_pickaxe": 1}
        _give_potions(self, count=1)
        _set_vitals(self, hp=9)
        _ensure_floor_gate_monsters_for(self, 1, self._EXIT_KILL_REQUIREMENT)
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        kills = self._floor_task_kills.get(1, 0)
        if kills >= 4:
            reward += self._stage_reward("four_kills", 0.25)
        if kills >= 8:
            reward += self._stage_reward("eight_kills", 0.5)
        if old_floor == 1 and self._current_floor == 2:
            reward += self._stage_reward("descended", 0.25)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxSpellOnlyKillEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Correct-element spell economy drill."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + ("floors:6",)
    _TARGET_TYPES = ("fire_elemental",)
    _TARGET_FLOOR = 6
    _KILL_TARGET = 2
    _GATED_FLOORS = (6,)

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-spell-only-kill-v0"

    def _task_description(self) -> str:
        return (
            "Defeat 2 fire elementals with spells. Your wood sword is nearly "
            "useless against their physical defense and fire immunity; use "
            "CAST_ICEBALL and kite while mana is finite. Reward: +0.5 per "
            "kill, -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 6)
        grid = self._floors[6]
        _clear_dungeon_area(grid, self._agent_x, self._agent_y, 5, _DUNGEON_SIZE)
        self._mobs = [m for m in self._mobs if m.get("floor") != 6]
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "fire_elemental", self._agent_x, self._agent_y, floor=6, radius=6)
        self._inventory = {"wood_sword": 1}
        self._learned_spells = {"fireball": False, "iceball": True}
        self._int_attr = 5
        self._recompute_max_stats()
        _set_vitals(self, hp=12, mana=self._max_mana)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = self._target_kills_this_step * (1.0 / self._KILL_TARGET)
        if self._target_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["target_kills"] = self._target_kills
        return obs, reward, terminated, truncated, info


class CraftaxDescentGauntletEnv(_FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Two-floor cumulative descent under a tuned local gate."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "crafting:torches", "floors:1", "floors:2", "floors:3",
        "floors:navigation",
    )
    _dungeon_ambient_light = 0.0
    _GATED_FLOORS = (1, 2)
    _EXIT_KILL_REQUIREMENT = 4

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-descent-gauntlet-v0"

    def _task_description(self) -> str:
        return (
            "Descend from floor 1 to floor 3 in one episode. To keep the "
            "<512-step cap, each floor uses a local 4-kill seal rather than "
            "copying the fork's 8-kill gate. Damage, potions, torches, mana, "
            "and HP carry across floors. Rewards: +0.25 for opening each "
            "floor seal and +0.25 for each DESCEND; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 1)
        self._inventory = {
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 12,
            "torches": 5,
            "wood": 4,
            "coal": 4,
        }
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_vitals(self, hp=12)
        for floor in self._GATED_FLOORS:
            _ensure_floor_gate_monsters_for(self, floor, self._EXIT_KILL_REQUIREMENT)
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        for floor in self._GATED_FLOORS:
            if self._floor_task_kills.get(floor, 0) >= self._EXIT_KILL_REQUIREMENT:
                reward += self._stage_reward(f"gate_{floor}", 0.25)
        if old_floor == 1 and self._current_floor == 2:
            reward += self._stage_reward("descend_2", 0.25)
        if old_floor == 2 and self._current_floor == 3:
            reward += self._stage_reward("descend_3", 0.25)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxPotionIdStakesEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Sparse potion-identification task where poison can kill."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS
    _TARGET_TYPES = ("zombie",)
    _TARGET_FLOOR = 0
    _GATED_FLOORS = (0,)

    _POTION_ACTION_TO_INDEX = CraftaxPotionTriageEnv._POTION_ACTION_TO_INDEX

    def __init__(self, max_turns: int = 80) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-potion-id-stakes-v0"

    def _task_description(self) -> str:
        return (
            "You are at 2 HP with only three hidden-color potions: one heal, "
            "one poison, and one harmful drain. Identify the heal before "
            "fighting the zombie. Rewards: +0.5 for drinking the heal potion, "
            "+0.5 for killing the zombie; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        from glyphbench.envs.craftax.mechanics.potions import POTION_EFFECTS

        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=5, scatter=8)
        self._mobs = []
        _place_full_mob(self, "zombie", cx, cy, floor=0, radius=5)
        colors = ("red", "green", "blue", "pink", "cyan", "yellow")
        wanted = {"heal_8", "poison_3", "mana_drain_3"}
        potions = {color: 0 for color in colors}
        for idx, effect_idx in enumerate(self._potion_mapping):
            if POTION_EFFECTS[effect_idx] in wanted:
                potions[colors[idx]] = 1
        self._inventory = {"wood_sword": 1, "potions": potions}
        _set_vitals(self, hp=2, mana=0, energy=4)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        from glyphbench.envs.craftax.mechanics.potions import POTION_EFFECTS

        action_name = self.action_spec.names[action]
        potion_idx = self._POTION_ACTION_TO_INDEX.get(action_name)
        potion_effect = None
        if potion_idx is not None:
            color = action_name.removeprefix("DRINK_POTION_").lower()
            if self._inventory.get("potions", {}).get(color, 0) > 0:
                potion_effect = POTION_EFFECTS[self._potion_mapping[potion_idx]]

        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if potion_effect == "heal_8":
            reward += self._stage_reward("heal", 0.5)
        if self._target_kills >= 1:
            reward += self._stage_reward("kill", 0.5)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxClearFloor8GateEnv(_FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Single-floor 8-kill gate with enough kit to fit the cap."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "floors:4", "floors:5", "floors:navigation",
    )
    _GATED_FLOORS = (4,)
    _EXIT_KILL_REQUIREMENT = 8

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-clear-floor-8gate-v0"

    def _task_description(self) -> str:
        return (
            "Clear the Vaults' local 8-kill stair seal with fast tools, then "
            "DESCEND. This isolates the true gate magnitude without a long "
            "full-game run. Rewards: +0.25 at 4 kills, +0.5 at 8 kills, "
            "+0.25 on DESCEND; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 4)
        self._inventory = {"diamond_sword": 1, "bow": 1, "arrows": 12}
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        _learn_all_spells(self)
        _give_potions(self, count=1)
        _set_vitals(self, hp=16)
        _ensure_floor_gate_monsters_for(self, 4, self._EXIT_KILL_REQUIREMENT)
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        kills = self._floor_task_kills.get(4, 0)
        if kills >= 4:
            reward += self._stage_reward("four_kills", 0.25)
        if kills >= 8:
            reward += self._stage_reward("eight_kills", 0.5)
        if old_floor == 4 and self._current_floor == 5:
            reward += self._stage_reward("descended", 0.25)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


def _stamp_surface_progression_resources(env: CraftaxFullEnv, cx: int, cy: int) -> None:
    grid = env._floors[0]
    layout = {
        (-5, -4): TILE_TREE, (-4, -4): TILE_TREE, (-3, -4): TILE_TREE,
        (-5, -2): TILE_TREE, (-4, -2): TILE_TREE,
        (3, -4): TILE_STONE, (4, -4): TILE_STONE, (5, -4): TILE_STONE,
        (3, -2): TILE_STONE, (5, -2): TILE_STONE,
        (-5, 3): TILE_COAL, (-3, 3): TILE_COAL, (-1, 4): TILE_COAL,
        (1, 4): TILE_COAL, (3, 3): TILE_COAL,
        (-4, 5): TILE_IRON, (-2, 5): TILE_IRON, (0, 6): TILE_IRON,
        (2, 5): TILE_IRON, (4, 5): TILE_IRON,
    }
    for (dx, dy), tile in layout.items():
        x, y = cx + dx, cy + dy
        if 1 <= x < _SURFACE_SIZE - 1 and 1 <= y < _SURFACE_SIZE - 1:
            grid[y][x] = tile


def _move_surface_entrance_near(env: CraftaxFullEnv, cx: int, cy: int) -> None:
    old = env._stairs_down_pos.get(0)
    if old is not None:
        ox, oy = old
        env._floors[0][oy][ox] = TILE_GRASS
    entrance = (min(_SURFACE_SIZE - 3, cx + 8), min(_SURFACE_SIZE - 3, cy + 7))
    ex, ey = entrance
    env._floors[0][ey][ex] = TILE_STAIRS_DOWN
    env._stairs_down_pos[0] = entrance
    _carve_surface_route(env, (cx, cy), entrance)


class CraftaxSurfaceToIronDescendEnv(
    _FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv
):
    """Earn iron gear, clear a short surface gate, and enter dungeon."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "crafting:wood", "crafting:stone", "crafting:iron",
        "survival:day_night", "floors:0", "floors:1",
    )
    _GATED_FLOORS = (0,)
    _EXIT_KILL_REQUIREMENT = 3

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-surface-to-iron-descend-v0"

    def _task_description(self) -> str:
        return (
            "Start empty-handed on the surface. Craft an iron pickaxe and iron "
            "sword, defeat the local 3-kill surface seal, then DESCEND into "
            "the dungeon. Rewards: +0.25 iron pickaxe, +0.25 iron sword, "
            "+0.25 seal opened, +0.25 on DESCEND; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=8, scatter=16)
        _stamp_surface_progression_resources(self, cx, cy)
        _move_surface_entrance_near(self, cx, cy)
        self._inventory = {}
        _set_vitals(self, hp=9)
        for _ in range(self._EXIT_KILL_REQUIREMENT):
            _spawn_surface_pressure_mob(self, "zombie", min_dist=7, max_dist=10)
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._inventory.get("iron_pickaxe", 0) > 0:
            reward += self._stage_reward("iron_pickaxe", 0.25)
        if self._inventory.get("iron_sword", 0) > 0:
            reward += self._stage_reward("iron_sword", 0.25)
        if self._floor_task_kills.get(0, 0) >= self._EXIT_KILL_REQUIREMENT:
            reward += self._stage_reward("gate", 0.25)
        if old_floor == 0 and self._current_floor == 1:
            reward += self._stage_reward("descend", 0.25)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxNightHordeNoShelterEnv(
    _TargetKillMixin, _ContinuousNightPressureMixin, _FocusedRewardMixin, CraftaxFullEnv
):
    """Stochastic night fight without the scripted-wave or walling crutch."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS + (
        "survival:day_night", "crafting:torches",
    )
    _TARGET_TYPES = ("zombie", "skeleton")
    _TARGET_FLOOR = 0
    _GATED_FLOORS = (0,)
    _NIGHT_SPAWN_INTERVAL = 10
    _NIGHT_SPAWN_LIMIT = 12

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-night-horde-no-shelter-v0"

    def _task_description(self) -> str:
        return (
            "Survive a stochastic night horde without enough stone to wall "
            "off the fight. Kite with a stone sword and limited torches. "
            "Rewards: +0.2 at 2 kills, +0.2 at 4 kills, +0.2 at 40 night "
            "steps, +0.2 at 80 night steps, +0.2 at dawn; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=7, scatter=18)
        self._mobs = []
        self._inventory = {"stone_sword": 1, "torches": 2, "stone": 1}
        _set_vitals(self, hp=9)
        self._day_counter = _DAY_LENGTH - 5
        self._day_night = "day"
        for mtype in ("zombie", "skeleton"):
            _spawn_surface_pressure_mob(self, mtype, min_dist=6, max_dist=9)
        self._agent_x, self._agent_y = cx, cy
        self._reset_target_kills()
        return self._render_current_observation()

    def _night_elapsed(self) -> int:
        return min(_NIGHT_LENGTH, max(0, self._day_counter - _DAY_LENGTH))

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_phase = self._day_night
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._target_kills >= 2:
            reward += self._stage_reward("two_kills", 0.2)
        if self._target_kills >= 4:
            reward += self._stage_reward("four_kills", 0.2)
        elapsed = self._night_elapsed()
        if elapsed >= 40:
            reward += self._stage_reward("survive_40", 0.2)
        if elapsed >= 80:
            reward += self._stage_reward("survive_80", 0.2)
        if old_phase == "night" and self._day_night == "day" and self._hp > 0:
            reward += self._stage_reward("dawn", 0.2)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["target_kills"] = self._target_kills
        return obs, reward, terminated, truncated, info


class CraftaxXpAttributeSpendEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Make XP spending load-bearing instead of decorative."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS + (
        "progression:xp", "progression:attributes",
    )
    _TARGET_TYPES = ("zombie", "knight")
    _TARGET_FLOOR = 0
    _GATED_FLOORS = (0,)

    def __init__(self, max_turns: int = 220) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-xp-attribute-spend-v0"

    def _task_description(self) -> str:
        return (
            "Kill weak mobs to earn XP, spend LEVEL_UP_STRENGTH twice, then "
            "defeat the final knight. Rewards: +0.25 for strength 2, +0.25 "
            "for strength 3, +0.5 for the final kill; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=6, scatter=12)
        self._mobs = []
        for _ in range(2):
            _place_full_mob(self, "zombie", cx, cy, floor=0, radius=6)
        self._inventory = {"stone_sword": 1}
        _set_vitals(self, hp=9)
        self._xp = 0
        self._str = 1
        self._dex = 1
        self._int_attr = 1
        self._recompute_max_stats()
        self._hp = self._max_hp
        self._xp_final_spawned = False
        self._xp_final_killed = False
        self._reset_target_kills()
        return self._render_current_observation()

    def _record_target_kill(self, mob: FullMob) -> None:
        super()._record_target_kill(mob)
        if mob.get("type") == "zombie":
            self._xp += 1
        elif mob.get("type") == "knight":
            self._xp_final_killed = True

    def _spawn_final_if_ready(self) -> None:
        if self._xp_final_spawned or self._str < 3:
            return
        _place_full_mob(self, "knight", self._agent_x, self._agent_y, floor=0, radius=7)
        self._xp_final_spawned = True

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        self._spawn_final_if_ready()
        if self._xp_final_spawned:
            obs = self._render_current_observation()
        reward = 0.0
        if self._str >= 2:
            reward += self._stage_reward("str2", 0.25)
        if self._str >= 3:
            reward += self._stage_reward("str3", 0.25)
        if self._xp_final_killed:
            reward += self._stage_reward("final", 0.5)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["xp"] = self._xp
        info["strength"] = self._str
        return obs, reward, terminated, truncated, info


class CraftaxBowKiteHardEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Bow-economy hard variant: ammo is scarce and must be crafted."""

    tutorial_sections = _SURFACE_COMBAT_SECTIONS + ("crafting:arrows",)
    _TARGET_TYPES = ("skeleton",)
    _TARGET_FLOOR = 0
    _KILL_TARGET = 5
    _GATED_FLOORS = (0,)

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-bow-kite-hard-v0"

    def _task_description(self) -> str:
        return (
            "Defeat 5 skeletons with a bow, but you start with only 6 arrows. "
            "Use the nearby table and resources to MAKE_ARROW mid-fight when "
            "ammo runs low. Reward: +0.2 per skeleton kill; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=7, scatter=18)
        _place_table_and_furnace_near(self, cx, cy)
        self._mobs = []
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "skeleton", cx, cy, floor=0, radius=8)
        self._inventory = {
            "bow": 1,
            "arrows": 6,
            "wood_sword": 1,
            "wood": 5,
            "stone": 5,
        }
        _set_vitals(self, hp=9)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = self._target_kills_this_step * (1.0 / self._KILL_TARGET)
        if self._target_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        info["target_kills"] = self._target_kills
        return obs, reward, terminated, truncated, info


class CraftaxEnchantChoiceEnv(_TargetKillMixin, _FocusedRewardMixin, CraftaxFullEnv):
    """Choose the correct enchantment table for fire-immune mobs."""

    tutorial_sections = _MAGIC_COMBAT_SECTIONS + (
        "items:gems", "floors:6",
    )
    _TARGET_TYPES = ("fire_elemental",)
    _TARGET_FLOOR = 6
    _KILL_TARGET = 2
    _GATED_FLOORS = (6,)

    def __init__(self, max_turns: int = 180) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-enchant-choice-v0"

    def _task_description(self) -> str:
        return (
            "Both fire and ice enchant tables are present. Fire elementals are "
            "fire-immune, so choose the ice table, ENCHANT_WEAPON with "
            "sapphire, then defeat 2 elementals. Rewards: +0.5 for the ice "
            "weapon enchant and +0.25 per kill; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        _move_player_to_floor_start(self, 6)
        grid = self._floors[6]
        _clear_dungeon_area(grid, self._agent_x, self._agent_y, 6, _DUNGEON_SIZE)
        grid[self._agent_y][self._agent_x + 4] = TILE_ENCHANT_FIRE
        grid[self._agent_y + 4][self._agent_x] = TILE_ENCHANT_ICE
        self._mobs = [m for m in self._mobs if m.get("floor") != 6]
        for _ in range(self._KILL_TARGET):
            _place_full_mob(self, "fire_elemental", self._agent_x, self._agent_y, floor=6, radius=8)
        self._inventory = {"diamond_sword": 1, "ruby": 1, "sapphire": 1}
        self._sword_enchantment = 0
        for slot in ("helmet", "chest", "legs", "boots"):
            self._armor_slots[slot] = 1
        _set_vitals(self, hp=12, mana=_MAX_MANA)
        self._reset_target_kills()
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        self._target_kills_this_step = 0
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._sword_enchantment == 2:
            reward += self._stage_reward("ice_enchant", 0.5)
        reward += self._target_kills_this_step * (0.5 / self._KILL_TARGET)
        if self._sword_enchantment == 2 and self._target_kills >= self._KILL_TARGET:
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info


class CraftaxDiamondGearDescendEnv(
    _FloorExitGateMixin, _FocusedRewardMixin, CraftaxFullEnv
):
    """Extend diamond bootstrap through gear crafting and dungeon entry."""

    tutorial_sections = _FLOOR_NAV_BASE_SECTIONS + (
        "crafting:diamond", "survival:day_night", "floors:0", "floors:1",
    )
    _GATED_FLOORS = (0,)
    _EXIT_KILL_REQUIREMENT = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-diamond-gear-descend-v0"

    def _task_description(self) -> str:
        return (
            "From a curated surface resource field, craft diamond pickaxe, "
            "diamond sword, and one diamond armor piece, clear a 2-kill "
            "surface seal, then DESCEND. Rewards: five +0.2 bins for those "
            "stages; -1 on death."
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _prepare_surface_arena(self, radius=8, scatter=14)
        _stamp_surface_progression_resources(self, cx, cy)
        for dx in (-6, -5, -4, -3, 3, 4, 5, 6, 0, 1):
            x = cx + dx
            y = cy - 6 if dx <= 1 else cy + 6
            if 1 <= x < _SURFACE_SIZE - 1 and 1 <= y < _SURFACE_SIZE - 1:
                self._floors[0][y][x] = TILE_DIAMOND
        _move_surface_entrance_near(self, cx, cy)
        self._inventory = {}
        _set_vitals(self, hp=9)
        for _ in range(self._EXIT_KILL_REQUIREMENT):
            _spawn_surface_pressure_mob(self, "zombie", min_dist=8, max_dist=10)
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, _reward, terminated, truncated, info = super()._step(action)
        reward = 0.0
        if self._inventory.get("diamond_pickaxe", 0) > 0:
            reward += self._stage_reward("diamond_pickaxe", 0.2)
        if self._inventory.get("diamond_sword", 0) > 0:
            reward += self._stage_reward("diamond_sword", 0.2)
        if any(tier >= 2 for tier in self._armor_slots.values()):
            reward += self._stage_reward("diamond_armor", 0.2)
        if self._floor_task_kills.get(0, 0) >= self._EXIT_KILL_REQUIREMENT:
            reward += self._stage_reward("gate", 0.2)
        if old_floor == 0 and self._current_floor == 1:
            reward += self._stage_reward("descend", 0.2)
            terminated = True
            info["subtask_success"] = True
        if self._hp <= 0:
            reward += self._DEATH_PENALTY
            terminated = True
            info["subtask_success"] = False
        return obs, reward, terminated, truncated, info

# Registration is handled in glyphbench.envs.craftax.__init__.
