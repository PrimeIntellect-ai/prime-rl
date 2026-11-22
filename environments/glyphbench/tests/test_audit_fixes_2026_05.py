"""Regression tests for the 2026-05 full-suite audit correctness fixes.

Covers (1) the BaseGlyphEnv cumulative-return clamp safety net, (2) the craftax
virtual action-dispatch fix (subclass _handle_* overrides are actually called),
(3) the craftax _handle_place_plant edge-of-arena IndexError fix, and (4) the
atari berzerk single-death-penalty fix.
"""
from __future__ import annotations

import random
import types

import pytest

from glyphbench.core import make_env


# ---------------------------------------------------------------- base clamp
def test_base_step_clamps_cumulative_return_to_unit_interval():
    """A per-step reward that would push the running sum out of [-1, 1] is
    clamped so the cumulative return can never leave the bound."""
    env = make_env("glyphbench/atari-berzerk-v0")
    env.reset(seed=0)
    # Force a single -3.0 step: the base must clamp the emitted reward to -1.0.
    env._game_step = types.MethodType(lambda self, a: (-3.0, True, {}), env)
    _, r, term, _, _ = env.step(0)
    assert r == pytest.approx(-1.0)

    env2 = make_env("glyphbench/atari-breakout-v0")
    env2.reset(seed=0)
    env2._game_step = types.MethodType(lambda self, a: (5.0, False, {}), env2)
    _, r1, _, _, _ = env2.step(0)
    _, r2, _, _, _ = env2.step(0)
    assert r1 == pytest.approx(1.0)   # clamped up-front
    assert r2 == pytest.approx(0.0)   # budget already spent


def test_full_games_opt_out_of_return_clamp():
    """craftaxfull / nethack-full report RAW unclamped cumulative reward; the
    scored craftax minigames keep the [-1,1] clamp."""
    from glyphbench.core.action import ActionSpec
    from glyphbench.core.base_env import BaseGlyphEnv
    from glyphbench.core.observation import GridObservation
    from glyphbench.core.registry import REGISTRY
    from glyphbench.envs.nethack.full import NetHackFullEnv

    if "glyphbench/craftaxfull-v0" in REGISTRY:
        assert REGISTRY["glyphbench/craftaxfull-v0"].clamp_episode_return is False
    assert NetHackFullEnv.clamp_episode_return is False
    # A scored minigame keeps the clamp.
    assert make_env("glyphbench/craftax-choptrees-v0").clamp_episode_return is True

    # Behavioural check of the gating via a tiny stub: with the clamp off, the
    # cumulative return accumulates past [-1, 1].
    class _Stub(BaseGlyphEnv):
        action_spec = ActionSpec(names=("NOOP",), descriptions=("x",))
        clamp_episode_return = False

        def _reset(self, seed):
            return GridObservation("g", "l", "h", "m")

        def _step(self, a):
            return GridObservation("g", "l", "h", "m"), 5.0, False, False, {}

        def _render_current_observation(self):
            return GridObservation("g", "l", "h", "m")

        def system_prompt(self):
            return "x"

        def env_id(self):
            return "stub"

    s = _Stub(max_turns=10)
    s.reset(seed=0)
    _, r1, *_ = s.step(0)
    _, r2, *_ = s.step(0)
    _, r3, *_ = s.step(0)
    # Not clamped: every +5 reward passes through and the return exceeds [-1,1].
    assert r1 == pytest.approx(5.0)
    assert r2 == pytest.approx(5.0)
    assert r3 == pytest.approx(5.0)
    assert s._episode_return == pytest.approx(15.0)

    # A default (clamped) env caps the same reward stream at +1 cumulative.
    class _ClampStub(_Stub):
        clamp_episode_return = True

    c = _ClampStub(max_turns=10)
    c.reset(seed=0)
    _, cr1, *_ = c.step(0)
    _, cr2, *_ = c.step(0)
    assert cr1 == pytest.approx(1.0)  # clamped up-front
    assert cr2 == pytest.approx(0.0)  # budget already spent
    assert c._episode_return == pytest.approx(1.0)


# ------------------------------------------------ craftax virtual dispatch
@pytest.mark.parametrize("env_id", [
    "glyphbench/craftax-choptrees-v0",
    "glyphbench/craftax-minestone-v0",
])
def test_craftax_subclass_handle_do_override_is_dispatched(env_id):
    """The DO action must resolve to the subclass override, not the base
    CraftaxFullEnv._handle_do (the dead-dispatch bug)."""
    env = make_env(env_id)
    do_fn = env._ACTION_DISPATCH["DO"]
    bound = getattr(env, do_fn.__name__)
    # bound.__func__ must be the subclass's own _handle_do, not the base one.
    assert bound.__func__ is not None
    assert "FullEnv" not in bound.__func__.__qualname__, (
        f"{env_id} DO still dispatches to {bound.__func__.__qualname__}"
    )


def test_craftax_choptrees_chopping_wood_rewards_progress():
    """Driving DO toward trees yields positive reward and advances the
    progress counter (previously dead: reward stayed 0.0)."""
    env = make_env("glyphbench/craftax-choptrees-v0")
    names = env.action_spec.names
    do = names.index("DO")
    moves = [names.index(m) for m in ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN")]
    env.reset(seed=1)
    rng = random.Random(0)
    total = 0.0
    for t in range(120):
        a = do if t % 2 == 0 else rng.choice(moves)
        _, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    assert total > 0.0, "chopping wood produced no reward (dead dispatch?)"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9


# ------------------------------------------- craftax place_plant edge fix
@pytest.mark.parametrize("env_id", [
    "glyphbench/craftax-craftchain-v0",
    "glyphbench/craftax-craftpickaxe-v0",
    "glyphbench/craftax-craftsword-v0",
])
def test_craftax_place_plant_does_not_crash_at_arena_edge(env_id):
    """PLACE_PLANT at the arena edge facing outward must not IndexError
    (was hardcoding fsize=64 against a 16x16 arena)."""
    env = make_env(env_id)
    env.reset(seed=0)
    n = env._floor_size()
    env._agent_x, env._agent_y = n - 1, n - 1
    for facing in [(1, 0), (0, 1)]:
        env._facing = facing
        # must return cleanly (0.0, out of bounds) rather than raise
        assert env._handle_place_plant() == 0.0


# ----------------------------------------------- berzerk single death pen
def test_berzerk_death_penalty_applied_at_most_once_per_step():
    """Co-located robot bullets + a robot on the player cell must add the
    death penalty only once, keeping the per-step reward >= -1.0."""

    env = make_env("glyphbench/atari-berzerk-v0")
    env.reset(seed=0)
    px, py = env._player_x, env._player_y
    # Place two robot bullets and one robot directly on the player cell.
    b1 = env._add_entity("bullet", "•", px, py)
    b1.data["owner"] = "robot"
    b1.dx = b1.dy = 0
    b2 = env._add_entity("bullet", "•", px, py)
    b2.data["owner"] = "robot"
    b2.dx = b2.dy = 0
    env._add_entity("robot", "R", px, py)
    # Inspect the raw game reward so the base return clamp cannot conceal
    # duplicate death penalties.
    reward, terminated, _ = env._game_step("NOOP")
    assert terminated
    assert reward == pytest.approx(-1.0)


# ----------------------------------------------- memento-f4 cue/door glyph
def test_memento_f4_stepping_on_start_room_cue_does_not_terminate():
    """Cue glyphs reuse the door codepoints; stepping onto a cue in the
    start/middle room (x < fog_room_x0) must not trigger the decision door."""
    from glyphbench.envs.minihack.memento import COLOR_DOOR_GLYPHS, MOVE_VECTORS

    env = make_env("glyphbench/minihack-memento-f4-v0")
    env.reset(seed=0)
    x0 = env._fog_room_x0
    right = next(n for n, v in MOVE_VECTORS.items() if v == (1, 0))
    ai = env.action_spec.names.index(right)
    for y in range(25):
        for x in range(1, x0):
            try:
                c = env._terrain_at(x, y)
            except Exception:
                continue
            if c in COLOR_DOOR_GLYPHS:
                env._player_pos = (x - 1, y)
                _, _, term, trunc, _ = env.step(ai)
                assert not term, "stepping onto a start-room cue ended the episode"
                return
    pytest.skip("no start-room cue cell found to probe")


# ----------------------------------------------- pong mandatory Step HUD
def test_pong_hud_has_step_indicator():
    env = make_env("glyphbench/atari-pong-v0")
    obs, _ = env.reset(seed=0)
    assert "Step:" in str(obs)


# ------------------------------------- plunder: skilled beats FIRE-spam trivial
def test_plunder_skilled_policy_beats_fire_spam():
    """Retune: pirates spread across columns and fall straight, so a static
    'spam FIRE' agent (the old dominant trivial policy) can no longer win.
    A skilled traverse-aim-dodge captain must clearly outscore it. Also guards
    that UP was dropped from the action space."""
    eid = "glyphbench/procgen-plunder-v0"
    names = make_env(eid).action_spec.names
    assert "UP" not in names  # trap action removed
    fire = names.index("FIRE")
    left, right, noop = (names.index(n) for n in ("LEFT", "RIGHT", "NOOP"))
    ay = 12  # agent row (GRID_H - 2)

    def ships(env, etype):
        return [(e.x, e.y) for e in env._entities if e.alive and e.etype == etype]

    def skilled(env):
        ax = env._agent_x
        pir, civ = ships(env, "pirate"), ships(env, "civilian")

        def ram(x):
            return any(px == x and py == ay - 1 for px, py in pir)

        if ram(ax):  # dodge an imminent ram into a clear adjacent column
            for d, act in ((-1, left), (1, right)):
                nx = ax + d
                if 0 <= nx < env.GRID_W and not ram(nx):
                    return act
            return noop
        col = [(px, py) for px, py in pir if px == ax and py <= ay - 2]
        if col:  # fire only if no civilian sits below the target pirate
            nearest = max(col, key=lambda p: p[1])
            if not any(cx == ax and cy > nearest[1] for cx, cy in civ):
                return fire
        for px, py in sorted(pir, key=lambda p: (-p[1], abs(p[0] - ax))):
            if py >= ay - 1:
                continue
            d = -1 if px < ax else (1 if px > ax else 0)
            if d == 0:
                return fire
            if not ram(ax + d):
                return left if d < 0 else right
        return noop

    def run(policy, seeds):
        totals = []
        for s in seeds:
            env = make_env(eid)
            env.reset(seed=s)
            total = 0.0
            for _ in range(env.max_turns):
                _, r, term, trunc, _ = env.step(policy(env))
                total += r
                if term or trunc:
                    break
            assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
            totals.append(total)
        return sum(totals) / len(totals)

    seeds = range(20)
    fire_mean = run(lambda env: fire, seeds)
    skilled_mean = run(skilled, seeds)
    assert fire_mean < 0.1, f"FIRE-spam still strong: {fire_mean}"
    assert skilled_mean > 0.5, f"skilled policy cannot win: {skilled_mean}"
    assert skilled_mean > fire_mean + 0.4


def test_ninja_skilled_policy_beats_blind_throw_right_cycle():
    """Retune: pits carved into the traverse row (plus a step cost + death
    penalty) defeat the old dominant trivial policy, a blind THROW,RIGHT
    2-cycle that used to win 300/300. That cycle now walks into a pit, gets
    trapped, and never reaches the goal, while a terrain-aware ninja that
    jumps the pits clears the level on most seeds."""
    eid = "glyphbench/procgen-ninja-v0"
    names = make_env(eid).action_spec.names
    right, jr, throw, noop = (names.index(n) for n in ("RIGHT", "JUMP_RIGHT", "THROW", "NOOP"))
    ground_y = 10  # H - 2

    def blind(env, t):
        return throw if t % 2 == 0 else right

    def skilled(env, t):
        ax, ay = env._agent_x, env._agent_y
        gx = next(
            x for y in range(env._world_h) for x in range(env._world_w)
            if env._world_at(x, y) == "G"
        )
        st = env_state.setdefault(id(env), {"air": 0})
        if env._jump_step >= 0 or not env._on_ground:
            if st["air"] > 0 and ax < gx:
                st["air"] -= 1
                return right
            return noop
        st["air"] = 0
        if env._world_at(ax + 1, ay) == "B":
            return throw
        if any(e.alive and e.etype == "enemy" and e.y == ay and e.x > ax
               for e in env._entities) and env._world_at(ax + 1, ground_y) != "·":
            return throw
        if env._world_at(ax + 1, ground_y) == "·":
            x, w = ax + 1, 0
            while env._world_at(x, ground_y) == "·" and w < 8:
                w += 1
                x += 1
            st["air"] = w
            return jr
        return right

    env_state: dict = {}

    def run(policy, seeds):
        wins = 0
        for s in seeds:
            env = make_env(eid)
            env.reset(seed=s)
            total = 0.0
            for t in range(env.max_turns):
                _, r, term, trunc, _ = env.step(policy(env, t))
                total += r
                if term and env._world_at(env._agent_x, env._agent_y) == "G":
                    wins += 1
                if term or trunc:
                    break
            assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
        return wins

    seeds = range(30)
    assert run(blind, seeds) == 0, "blind THROW,RIGHT cycle still reaches the goal"
    assert run(skilled, seeds) >= 24, "terrain-aware ninja can no longer win"


# ------------------- craftax floor-gate family retune (was unsolvable/trivial)
def test_craftax_floorgate_family_retune():
    """One guard for the 2026-05 craftax floor-gate-family retune:
    (1) the floor-exit kill gate is a small count again (was 8 -> unsolvable at
        9 HP) and floor-nav HP is raised above the base 9;
    (2) a trivial always-DO policy still loses the gated dungeon, bounded [-1,1];
    (3) find-diamond's guaranteed diamond is seed-randomized (the fixed-10-east
        open-loop [RIGHT*9, DO] exploit no longer wins);
    (4) reach-dungeon's staircase is reachable on every probed seed and the
        spawn->entrance offset varies per seed (was a fixed +8,+8)."""
    from collections import deque

    from glyphbench.envs.craftax import subtasks_extended as se
    from glyphbench.envs.craftaxfull.full import SURFACE_WALKABLE

    # (1) + (2) gate count, raised HP, trivial loses + bound
    assert se._FLOOR_GATE_KILLS <= 4, "gate kill count regressed back up"
    env = make_env("glyphbench/craftax-dungeon-v0")
    env.reset(seed=0)
    assert env._max_hp > 9, "floor-nav HP budget was not raised"
    do = env.action_spec.names.index("DO")
    total = 0.0
    for _ in range(env.max_turns):
        _, r, term, trunc, _ = env.step(do)
        total += r
        if term or trunc:
            break
    assert total <= 0.0, "trivial always-DO won the gated dungeon"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9

    # (3) find-diamond open-loop exploit defeated + randomized placement
    fd = "glyphbench/craftax-find-diamond-v0"
    names = make_env(fd).action_spec.names
    right, do = names.index("MOVE_RIGHT"), names.index("DO")
    fd_offsets, fd_wins = set(), 0
    for s in range(20):
        env = make_env(fd)
        env.reset(seed=s)
        dp = env._diamond_pos
        fd_offsets.add((dp[0] - env._agent_x, dp[1] - env._agent_y))
        for a in ([right] * 9 + [do]) * 3:
            _, r, term, trunc, info = env.step(a)
            if term or trunc:
                fd_wins += int(info.get("subtask_success", False))
                break
    assert fd_wins == 0, "fixed [RIGHT*9, DO] open-loop script still wins"
    assert len(fd_offsets) >= 15, "diamond placement is not seed-randomized"

    # (4) reach-dungeon reachability + randomized spawn offset
    rd = "glyphbench/craftax-reach-dungeon-v0"
    rd_offsets, blocked = set(), 0
    for s in range(60):
        env = make_env(rd)
        env.reset(seed=s)
        grid = env._floors[0]
        size = len(grid)
        start = (env._agent_x, env._agent_y)
        goal = env._stairs_down_pos[0]
        rd_offsets.add((goal[0] - start[0], goal[1] - start[1]))
        q, seen, ok = deque([start]), {start}, False
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                ok = True
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                nc = (nx, ny)
                if (0 <= nx < size and 0 <= ny < size and nc not in seen
                        and (grid[ny][nx] in SURFACE_WALKABLE or nc == goal)):
                    seen.add(nc)
                    q.append(nc)
        blocked += int(not ok)
    assert blocked == 0, f"{blocked}/60 reach-dungeon seeds unreachable"
    assert len(rd_offsets) >= 30, "spawn->entrance offset is not seed-randomized"


def test_starpilot_skilled_policy_beats_fire_spam():
    """Retune: a stationary 'FIRE every tick' agent used to score median +1.0.
    Bullets now travel one cell/tick (no parity tunnelling), have a finite
    range and a fire cooldown, and enemies are spread across lanes, so a parked
    spammer only clears its own short zone and cannot bank a win. A mobile
    dodge-and-snipe policy must clearly outscore it, bounded in [-1, 1]."""
    import copy

    eid = "glyphbench/procgen-starpilot-v0"
    env = make_env(eid)
    names = env.action_spec.names
    fire = names.index("FIRE")
    assert env.max_turns < 512  # budget invariant preserved

    def safe(e, depth=4):  # greedy pure-dodge survival rollout
        e = copy.deepcopy(e)
        for _ in range(depth):
            ax, ay, H = e._agent_x, e._agent_y, e.GRID_H
            ens = [x for x in e._entities if x.alive and x.etype == "enemy"]

            def near(r, ax=ax, ens=ens):
                return min(
                    [x.x - ax for x in ens if x.y == r and x.x >= ax], default=999
                )

            opts = [(0, ay)]
            if ay - 1 >= 1:
                opts.append((names.index("UP"), ay - 1))
            if ay + 1 < H - 1:
                opts.append((names.index("DOWN"), ay + 1))
            _, _, term, trunc, _ = e.step(max(opts, key=lambda o: near(o[1]))[0])
            if term:
                return 0
            if trunc:
                return 1
        return 1

    def planner(env):  # 1-step lookahead with survival rollout
        best, best_s = 0, -1e9
        for name, a in ((n, i) for i, n in enumerate(names)):
            e = copy.deepcopy(env)
            k0, p0 = e._enemies_killed, e._powerups_collected
            _, _, term, _, _ = e.step(a)
            if term:
                s = -100.0
            else:
                gained = (e._enemies_killed - k0) + (e._powerups_collected - p0)
                surv = safe(e)
                ax, ay = e._agent_x, e._agent_y
                ens = [x for x in e._entities
                       if x.alive and x.etype == "enemy" and x.x >= ax]
                align = -0.05 * abs(min(ens, key=lambda x: x.x).y - ay) if ens else 0.0
                s = (gained * 3.0 + surv + align + (0.02 if name == "FIRE" else 0.0)
                     - (50.0 if surv == 0 else 0.0))
            if s > best_s:
                best_s, best = s, a
        return best

    def run(policy, seeds):
        totals = []
        for s in seeds:
            env = make_env(eid)
            env.reset(seed=s)
            total = 0.0
            for _ in range(env.max_turns):
                _, r, term, trunc, _ = env.step(policy(env))
                total += r
                if term or trunc:
                    break
            assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
            totals.append(total)
        return sum(totals) / len(totals)

    seeds = range(12)
    fire_mean = run(lambda env: fire, seeds)
    skilled_mean = run(planner, seeds)
    assert fire_mean < 0.2, f"FIRE-spam still strong: {fire_mean}"
    assert skilled_mean > 0.6, f"skilled policy cannot win: {skilled_mean}"
    assert skilled_mean > fire_mean + 0.5


def test_mirrorlaser_blind_scan_loses_skilled_solver_wins():
    """Retune: puzzles were solvable by a single mirror toggle 74% of the time,
    so a physics-blind 'rotate every mirror once in reading order' scanner won
    ~70% of episodes. The generator now requires a minimal solving set of >= 3
    toggles AND rejects any row-major prefix solution, so the blind scanner is
    defeated while a beam-tracing oracle still solves every seed in budget."""
    from glyphbench.envs.classics.mirror_laser import (
        GRID_SIZE,
        MIN_SOLUTION_TOGGLES,
        MIRROR_BWD,
        MIRROR_FWD,
    )

    eid = "glyphbench/classics-mirrorlaser-v0"

    def mirror_cells(env):
        return [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)
                if env._mirrors[y][x] is not None]

    def min_solving_mask(env):
        cells = mirror_cells(env)
        n = len(cells)
        orig = [env._mirrors[y][x] for (x, y) in cells]
        best, best_mask = None, None
        for mask in range(1 << n):
            cnt = bin(mask).count("1")
            if best is not None and cnt >= best:
                continue
            for i, (x, y) in enumerate(cells):
                env._mirrors[y][x] = (
                    (MIRROR_BWD if orig[i] == MIRROR_FWD else MIRROR_FWD)
                    if (mask >> i) & 1 else orig[i]
                )
            if env._beam_hits_target(env._mirrors):
                best, best_mask = cnt, mask
        for i, (x, y) in enumerate(cells):
            env._mirrors[y][x] = orig[i]
        return cells, best, best_mask

    def blind_scan(seed):
        env = make_env(eid)
        env.reset(seed=seed)
        total = 0.0
        for (x, y) in mirror_cells(env):
            _, r, term, trunc, _ = env.step(y * GRID_SIZE + x)
            total += r
            if term or trunc:
                break
        assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
        return total

    def oracle(seed):
        env = make_env(eid)
        env.reset(seed=seed)
        cells, mt, mask = min_solving_mask(env)
        assert mt is not None and mt >= MIN_SOLUTION_TOGGLES, f"trivial puzzle seed {seed}"
        assert mt <= env.max_turns, f"min solution exceeds budget seed {seed}"
        total = 0.0
        for i in range(len(cells)):
            if (mask >> i) & 1:
                x, y = cells[i]
                _, r, term, trunc, _ = env.step(y * GRID_SIZE + x)
                total += r
                if term or trunc:
                    break
        assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
        return total

    seeds = range(40)
    blind_wins = sum(1 for s in seeds if blind_scan(s) > 0)
    oracle_wins = sum(1 for s in seeds if oracle(s) > 0)
    assert blind_wins <= 6, f"blind reading-order scan still wins {blind_wins}/40"
    assert oracle_wins == 40, f"beam-tracing oracle solves only {oracle_wins}/40"


def test_platformer_flag_always_reachable_after_regen_gate():
    """Retune: the flag was embedded inside a platform on ~7.8% of seeds and
    geometrically unreachable on a further ~2.3% (~10.6% unwinnable total). A
    flag-cell clearing pass plus an end-of-generation BFS reachability gate now
    guarantees the win condition is satisfiable on every seed, while the level
    layout (and thus the skill challenge) is otherwise untouched."""
    env = make_env("glyphbench/classics-platformer-v0")
    for s in range(120):
        env.reset(seed=s)
        # flag cell itself must be clear...
        assert not env._is_solid(env._flag_x, env._flag_y), f"flag embedded, seed {s}"
        # ...and a perfect agent must have a physical route to it.
        assert env._flag_is_reachable(), f"flag BFS-unreachable, seed {s}"


def test_wavedefense_threat_oracle_wins_fire_spam_loses():
    """Retune: the original tuning (enemies advance every turn, spawn every 3)
    was unwinnable on ~95% of seeds (best achievable return negative). Slowing
    enemy advance to once every ENEMY_MOVE_PERIOD turns and spawning no more
    often than SPAWN_INTERVAL gives the once-per-turn cannon a fair economy: a
    threat-priority oracle now wins most seeds with clearly positive return,
    while FIRE-only / random spam still lose every seed."""
    import math

    from glyphbench.envs.classics.wave_defense import (
        CENTER,
        DIR_DELTAS,
        DIRECTIONS,
    )

    eid = "glyphbench/classics-wavedefense-v0"

    def ray_dir(ex, ey):
        dx, dy = ex - CENTER, ey - CENTER
        for d, (ddx, ddy) in DIR_DELTAS.items():
            if ddx == 0 and dx == 0 and dy * ddy > 0:
                return d
            if ddy == 0 and dy == 0 and dx * ddx > 0:
                return d
            if ddx and ddy and dx * ddx > 0 and dy * ddy > 0 and abs(dx) == abs(dy):
                return d
        return None

    def chev(e):
        return max(abs(e[0] - CENTER), abs(e[1] - CENTER))

    def oracle_action(env):
        enemies = env._enemies
        if not enemies:
            return 3  # WAIT
        cur = env._facing_idx
        on_ray = sorted(
            ((chev(e), ray_dir(*e)) for e in enemies if ray_dir(*e) is not None),
            key=lambda x: x[0],
        )
        if on_ray:
            d = on_ray[0][1]
            if DIRECTIONS[cur] == d:
                return 2  # FIRE
            diff = (DIRECTIONS.index(d) - cur) % 8
            return 0 if diff <= 4 else 1
        tgt = min(enemies, key=chev)
        oc = int(round((math.degrees(math.atan2(tgt[0] - CENTER, -(tgt[1] - CENTER))) % 360) / 45)) % 8
        if oc == cur:
            return 3
        diff = (oc - cur) % 8
        return 0 if diff <= 4 else 1

    def run(policy):
        wins, total = 0, 0.0
        for s in range(40):
            env = make_env(eid)
            env.reset(seed=s)
            ep = 0.0
            outcome = ""
            for _ in range(env.max_turns):
                _, r, term, trunc, info = env.step(policy(env))
                ep += r
                assert -1.0 - 1e-9 <= ep <= 1.0 + 1e-9, (s, ep)
                if term or trunc:
                    outcome = info.get("outcome", "")
                    break
            total += ep
            wins += outcome == "victory"
        return wins, total / 40

    oracle_wins, oracle_mean = run(oracle_action)
    fire_wins, fire_mean = run(lambda e: 2)
    assert oracle_wins >= 24, f"threat oracle only wins {oracle_wins}/40"
    assert oracle_mean > 0.2, f"threat oracle mean return {oracle_mean:+.3f} not clearly positive"
    assert fire_wins == 0, f"FIRE-spam wins {fire_wins}/40 (should be unwinnable trivially)"
    assert fire_mean < -0.5, f"FIRE-spam mean {fire_mean:+.3f} not clearly sub-optimal"


@pytest.mark.parametrize(
    "eid,floor",
    [
        ("glyphbench/classics-rushhour-easy-v0", 4),
        ("glyphbench/classics-rushhour-hard-v0", 8),
    ],
)
def test_rushhour_difficulty_floor_kills_forward_only_policy(eid, floor):
    """Retune: the generator only checked BFS-solvability, so >50% of easy (and
    ~26% of hard) boards had an already-clear exit lane that a trivial
    always-MOVE_PLAYER_FWD policy solved without reasoning. We now require the
    exit lane to be blocked AND the BFS-optimal solution depth to meet a floor
    (>=4 easy, >=8 hard). Every generated board must meet the floor, and a
    forward-only policy must win 0 seeds (it can never clear its own blocker)."""
    from glyphbench.envs.classics.rush_hour import _bfs_optimal_depth

    fwd_wins = 0
    for s in range(30):
        env = make_env(eid)
        env.reset(seed=s)
        depth = _bfs_optimal_depth([v.copy() for v in env._vehicles])
        assert depth >= floor, f"seed {s}: optimal depth {depth} below floor {floor}"
        fwd = env.action_spec.names.index("MOVE_PLAYER_FWD")
        ep = 0.0
        won = False
        for _ in range(env.max_turns):
            _, r, term, trunc, _info = env.step(fwd)
            ep += r
            assert -1.0 - 1e-6 <= ep <= 1.0 + 1e-6, (s, ep)
            if term or trunc:
                won = r > 0
                break
        fwd_wins += won
    assert fwd_wins == 0, f"{eid}: forward-only policy won {fwd_wins}/30 (floor too weak)"


def test_climber_goal_always_reachable_after_regen_gate():
    """Retune: the goal `G` was geometrically unreachable on ~20% of seeds
    (top platform out of the 3-cell jump reach, or a mid-level horizontal
    dead-end), capping max return below the advertised 1.0 regardless of skill.
    `_generate_level` now re-rolls the layout (deterministically, per seed)
    until a faithful physics BFS confirms `G` is reachable, so a skilled agent
    always has a route to the goal while trivial policies still score ~0."""
    env = make_env("glyphbench/procgen-climber-v0")
    for s in range(80):
        env.reset(seed=s)
        assert env._goal_is_reachable(), f"goal BFS-unreachable, seed {s}"
    # Trivial single-action spam must remain clearly sub-optimal (never wins).
    for action in (env.action_spec.names.index("JUMP"),
                   env.action_spec.names.index("JUMP_RIGHT")):
        for s in range(20):
            env.reset(seed=s)
            ep = 0.0
            for _ in range(env.max_turns):
                _, r, term, trunc, _info = env.step(action)
                ep += r
                assert -1.0 - 1e-6 <= ep <= 1.0 + 1e-6, (s, ep)
                if term or trunc:
                    break
            assert ep < 0.2, f"trivial action {action} scored {ep} on seed {s}"


# ------------------------------------------------- floodfill-hard budget retune
def test_floodfill_hard_budget_discriminates():
    """Retune: the hard budget was lowered 35 -> 27. At 35 a state-blind
    color-cycle policy cleared the board ~76% of the time; at 27 it collapses
    to well under a third while a 1-step-lookahead greedy policy still wins the
    large majority of seeds. Guards both the budget value and the separation."""
    from collections import deque

    NUM_COLORS = 6
    env = make_env("glyphbench/classics-floodfill-hard-v0")
    assert env.max_turns == 27

    def _region(board, rows, cols):
        color = board[0][0]
        seen = {(0, 0)}
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in seen and board[nr][nc] == color):
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen

    def _greedy(e):
        board = [row[:] for row in e._board]
        rows, cols = e._grid_rows, e._grid_cols
        region = _region(board, rows, cols)
        cur = board[0][0]
        best, best_gain = 0, -1
        for nc in range(NUM_COLORS):
            if nc == cur:
                continue
            tmp = [row[:] for row in board]
            for (r, c) in region:
                tmp[r][c] = nc
            gain = len(_region(tmp, rows, cols)) - len(region)
            if gain > best_gain:
                best_gain, best = gain, nc
        return best

    n = 40
    cycle_wins = greedy_wins = 0
    for s in range(n):
        # state-blind cycle
        env.reset(seed=s)
        ep = 0.0
        for t in range(env.max_turns):
            _, r, term, trunc, _ = env.step(t % NUM_COLORS)
            ep += r
            assert -1.0 - 1e-6 <= ep <= 1.0 + 1e-6, (s, ep)
            if term:
                cycle_wins += 1
            if term or trunc:
                break
        # greedy 1-step lookahead
        env.reset(seed=s)
        ep = 0.0
        for _ in range(env.max_turns):
            _, r, term, trunc, _ = env.step(_greedy(env))
            ep += r
            assert -1.0 - 1e-6 <= ep <= 1.0 + 1e-6, (s, ep)
            if term:
                greedy_wins += 1
            if term or trunc:
                break

    assert cycle_wins <= 0.35 * n, f"blind cycle won {cycle_wins}/{n} (budget too easy)"
    assert greedy_wins >= 0.85 * n, f"greedy only won {greedy_wins}/{n} (budget too tight)"


def test_freeway_updown_spam_loses_skilled_dodger_wins():
    """Retune: a collision now resets the chicken to the start of the current
    crossing (was a free 2-row bump) and the budget is tightened 200 -> 60, so
    a blind UP-spam (which used to win 100/100) is now caught and runs out of
    time on most seeds, while a timing-aware dodge-and-wait policy still crosses
    4 times on every seed. Bounded in [0, 1]."""
    eid = "glyphbench/miniatari-freeway-v0"
    env = make_env(eid)
    assert env.max_turns == 60 and env.max_turns < 512
    up = env.action_spec.names.index("UP")
    noop = env.action_spec.names.index("NOOP")

    def _safe_to_enter(e, ty):
        px = e._player_x
        if ty not in e._LANE_ROWS:
            return True
        li = e._LANE_ROWS.index(ty)
        now = set(e._cars[li])
        d = e._lane_dirs[li]
        after = {(c + d) % e._WIDTH for c in e._cars[li]}
        return px not in now and px not in after

    def skilled(e):  # greedy: climb when the cell above is clear, else wait
        ty = e._player_y - 1
        return up if (ty < 0 or _safe_to_enter(e, ty)) else noop

    def run(policy, seeds):
        wins = 0
        for s in seeds:
            e = make_env(eid)
            e.reset(seed=s)
            ep = 0.0
            won = False
            for _ in range(e.max_turns):
                _, r, term, trunc, info = e.step(policy(e))
                ep += r
                assert -1.0 - 1e-9 <= ep <= 1.0 + 1e-9, (s, ep)
                if term or trunc:
                    won = bool(info.get("won", False))
                    break
            wins += int(won)
        return wins

    seeds = range(30)
    up_wins = run(lambda e: up, seeds)
    skilled_wins = run(skilled, seeds)
    assert up_wins <= 12, f"UP-spam still wins {up_wins}/30 (collisions not costly enough)"
    assert skilled_wins >= 28, f"skilled dodger only wins {skilled_wins}/30 (budget too tight)"


def test_icesliding_medium_random_loses_bfs_planner_wins():
    """Audit 2026-05 retune of classics-icesliding-medium-v0.

    BEFORE: budget=150 vs mean optimal len 3.4 let a uniform-random slider win
    70% with positive mean; ~45% of seeds were solvable in <=2 slides. The fix
    tightens the budget to 30 and rejects puzzles with BFS-optimal length < 3.
    A genuine planner (BFS shortest slide path) must still win; random must not.
    """
    from collections import deque

    eid = "glyphbench/classics-icesliding-medium-v0"
    env = make_env(eid)
    assert env.max_turns == 30
    deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def bfs_first_action(e):
        start, goal = e._player, e._goal
        prev: dict = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for a, (dx, dy) in deltas.items():
                n = e._slide_result(c[0], c[1], dx, dy)
                if n == c or n in e._trapdoors or n in prev:
                    continue
                prev[n] = (c, a)
                q.append(n)
        if goal not in prev:
            return 0
        cur, first = goal, 0
        while prev[cur] is not None:
            parent, a = prev[cur]
            first, cur = a, parent
        return first

    seeds = range(40)
    rand_wins = bfs_wins = 0
    for s in seeds:
        env.reset(seed=s)
        rng = random.Random(s * 7919 + 1)
        ep = 0.0
        for _ in range(env.max_turns):
            _, r, term, trunc, info = env.step(rng.randrange(4))
            ep += r
            assert -1.0 <= ep <= 1.0, (s, ep)
            if term or trunc:
                break
        rand_wins += int(info.get("goal_reached", False))

        env.reset(seed=s)
        ep = 0.0
        for _ in range(env.max_turns):
            _, r, term, trunc, info = env.step(bfs_first_action(env))
            ep += r
            assert -1.0 <= ep <= 1.0, (s, ep)
            if term or trunc:
                break
        bfs_wins += int(info.get("goal_reached", False))

    assert rand_wins <= 0.35 * 40, f"random won {rand_wins}/40 (budget too generous)"
    assert bfs_wins == 40, f"BFS planner only won {bfs_wins}/40 (not winnable)"


def test_amidar_seed_randomized_kills_fixed_script_optimal_planner_wins():
    """Retune: amidar _generate_level now consumes self.rng to randomize the
    rectangle origin, player spawn, and patrol start/direction per seed (was
    byte-identical every seed). The audit's dominant fixed 11-action script that
    won 30/30 now wins almost none, while an optimal joint-state planner (the
    patrol is fully deterministic) still wins every seed -> winnable +
    discriminative. Bounded in [-1, 1]."""
    from collections import deque

    eid = "glyphbench/miniatari-amidar-v0"
    env = make_env(eid)
    assert env.max_turns == 300 and env.max_turns < 512
    names = env.action_spec.names
    idx = {n: i for i, n in enumerate(names)}

    # (1) seed-sensitivity: distinct initial renders across seeds.
    renders = {make_env(eid).reset(seed=s)[0] for s in range(30)}
    assert len(renders) >= 25, f"only {len(renders)}/30 distinct renders (seed collapse)"

    # (2) the audit's dominant fixed script must no longer dominate.
    script = ["NOOP", "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "UP", "UP", "UP", "LEFT"]
    script_wins = 0
    for s in range(30):
        e = make_env(eid)
        e.reset(seed=s)
        ep = 0.0
        won = False
        for a in script:
            _, r, term, trunc, info = e.step(idx[a])
            ep += r
            assert -1.0 - 1e-9 <= ep <= 1.0 + 1e-9, (s, ep)
            if term or trunc:
                won = bool(info.get("won", False))
                break
        script_wins += int(won)
    assert script_wins <= 5, f"fixed script still wins {script_wins}/30 (layout not randomized enough)"

    # (3) optimal planner over the deterministic joint state wins every seed.
    def solvable(seed: int) -> bool:
        e = make_env(eid)
        e.reset(seed=seed)
        cells = e._perimeter_cells()
        ncells = len(cells)
        pdir = e._patrol_dir
        w, h = e._WIDTH, e._HEIGHT
        start = ((e._player_x, e._player_y), frozenset(cells), 0, e._patrol_step)
        seen = {start}
        q = deque([start])
        while q:
            (px, py), unp, tick, pstep = q.popleft()
            if not unp:
                return True
            if tick >= e.max_turns:
                continue
            ntick = tick + 1
            npstep = (pstep + pdir) % ncells if ntick % 3 == 0 else pstep
            pat = cells[npstep]
            for nm in names:
                nx, ny = px, py
                if nm == "LEFT":
                    nx = max(0, px - 1)
                elif nm == "RIGHT":
                    nx = min(w - 1, px + 1)
                elif nm == "UP":
                    ny = max(0, py - 1)
                elif nm == "DOWN":
                    ny = min(h - 1, py + 1)
                nunp = unp - {(nx, ny)} if (nx, ny) in unp else unp
                if not nunp:
                    return True
                if (nx, ny) == pat:
                    continue
                ns = ((nx, ny), nunp, ntick, npstep)
                if ns not in seen:
                    seen.add(ns)
                    q.append(ns)
        return False

    assert all(solvable(s) for s in range(30)), "some seeds unwinnable for an optimal planner"
