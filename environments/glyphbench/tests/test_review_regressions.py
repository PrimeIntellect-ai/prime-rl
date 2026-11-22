from __future__ import annotations

import pytest


def _a(env, name: str) -> int:
    return env.action_spec.names.index(name)


@pytest.mark.parametrize("game", ["robotank", "solaris"])
def test_atari_reset_clears_episode_progress_and_firing_direction(game: str) -> None:
    from glyphbench.core import make_env

    env = make_env(f"glyphbench/atari-{game}-v0")
    initial, _ = env.reset(seed=1729)
    first_shot = env.step(_a(env, "FIRE"))
    env.step(_a(env, "RIGHT"))
    if game == "robotank":
        env._kills = 4
    else:
        env.step(_a(env, "WARP"))
        env._sectors_cleared.add(0)

    reset_obs, _ = env.reset(seed=1729)

    assert reset_obs == initial
    assert env.step(_a(env, "FIRE")) == first_shot


def test_demonattack_same_step_kill_and_bomb_hit_preserves_kill_reward() -> None:
    from glyphbench.envs.miniatari.demonattack import MiniDemonAttackEnv

    env = MiniDemonAttackEnv()
    env.reset(seed=0)
    env._demons = [[env._player_x, 3, 1]]
    env._bombs = [[env._player_x, 8]]

    _, reward, terminated, _, _ = env.step(_a(env, "FIRE"))

    assert terminated
    assert reward == pytest.approx(-0.8)


def test_enduro_fatal_collision_tick_keeps_overtake_reward() -> None:
    from glyphbench.envs.miniatari.enduro import MiniEnduroEnv

    env = MiniEnduroEnv()
    env.reset(seed=0)
    env._lane_idx = 1
    env._player_x = env._LANES[1]
    env._collisions = 1
    env._cars = [[1, 7], [0, 7]]

    _, reward, terminated, _, _ = env.step(_a(env, "ACCEL"))

    assert terminated
    assert reward == pytest.approx(env._OVERTAKE_REWARD)


def test_enduro_dual_overtake_finish_stays_within_cap() -> None:
    from glyphbench.envs.miniatari.enduro import MiniEnduroEnv

    env = MiniEnduroEnv()
    env.reset(seed=0)
    env._lane_idx = 1
    env._player_x = env._LANES[1]
    env._progress = 4
    env._collisions = 0
    env._score = 4 * env._OVERTAKE_REWARD
    env._cars = [[0, 7], [2, 7]]

    _, reward, terminated, _, info = env.step(_a(env, "ACCEL"))

    assert terminated
    assert reward == pytest.approx(env._OVERTAKE_REWARD + env._NO_DAMAGE_BONUS)
    assert info["score"] == pytest.approx(1.0)
    assert info["progress"] == env._WIN_TARGET


def test_beamrider_same_step_kill_and_collision_is_additive() -> None:
    from glyphbench.envs.miniatari.beamrider import MiniBeamRiderEnv

    env = MiniBeamRiderEnv()
    env.reset(seed=0)
    env._beam_idx = 2
    env._player_x = env._BEAMS[2]
    env._progress = 0
    env._tick_count = 1
    env._fire_cd = 0
    env._enemies = [[2, 10, 1], [2, 10, 1]]

    _, reward, terminated, _, _ = env.step(_a(env, "FIRE"))

    assert terminated
    assert reward == pytest.approx(env._progress_reward(env._WIN_TARGET) - 1.0)


def test_bomberman_death_step_preserves_crate_reward() -> None:
    from glyphbench.envs.classics.bomberman import CRATE, FLOOR, WALL, BombermanEnv

    env = BombermanEnv()
    env.reset(seed=0)
    size = len(env._grid)
    env._grid = [[FLOOR] * size for _ in range(size)]
    for i in range(size):
        env._grid[0][i] = WALL
        env._grid[size - 1][i] = WALL
        env._grid[i][0] = WALL
        env._grid[i][size - 1] = WALL
    env._grid[1][2] = CRATE
    env._total_crates = 1
    env._bombs = [[1, 1, 1]]

    _, reward, terminated, _, info = env.step(_a(env, "WAIT"))

    assert terminated
    assert info["dead"] is True
    assert reward == pytest.approx(-0.5)


def test_craftax_bossfight_floor_boss_does_not_despawn() -> None:
    from glyphbench.envs.craftax.subtasks_extended import CraftaxBossFightEnv

    env = CraftaxBossFightEnv()
    env.reset(seed=42)
    boss = next(m for m in env._mobs if m.get("is_boss") and m.get("floor") == 5)
    boss["x"] = min(env._agent_x + 20, 30)
    boss["y"] = env._agent_y

    env.step(_a(env, "NOOP"))

    assert boss in env._mobs
    assert env._bosses_alive[5] is True


def test_craftax_firstday_uses_surface_achievement_denominator() -> None:
    from glyphbench.envs.craftax.subtasks_extended import CraftaxFirstDayEnv

    env = CraftaxFirstDayEnv()
    obs, _ = env.reset(seed=0)

    assert len(env._ALL_ACHIEVEMENTS) == 22
    assert "Achievements:" in obs
    assert "(0/22)" in obs
    assert env._try_unlock("collect_wood") == pytest.approx(1 / 22)
    assert env._try_unlock("collect_wood") == 0.0
    assert env._try_unlock("full_health") == 0.0


def test_craftax_survive_wild_rewards_survival_milestones() -> None:
    from glyphbench.envs.craftax.subtasks_extended import CraftaxSurviveWildEnv

    env = CraftaxSurviveWildEnv()
    env.reset(seed=0)

    rewards = []
    for _ in range(env._SURVIVE_MILESTONE_STEPS):
        _, reward, terminated, _, _ = env.step(_a(env, "NOOP"))
        rewards.append(reward)
        assert not terminated

    assert sum(rewards[:-1]) == pytest.approx(0.0)
    assert rewards[-1] == pytest.approx(0.1)


def test_craftax_wave_defense_uses_meaningful_kill_target() -> None:
    from glyphbench.envs.craftax.subtasks_extended import CraftaxWaveDefenseEnv

    assert CraftaxWaveDefenseEnv._WAVE_SIZES == (3, 3, 4)
    assert CraftaxWaveDefenseEnv._KILL_TARGET == 10
    assert pytest.approx(0.1) == 1.0 / CraftaxWaveDefenseEnv._KILL_TARGET


def test_boxoban_first_target_fill_gets_partial_reward() -> None:
    from glyphbench.envs.minihack.boxoban import MiniHackBoxobanUnfilteredEnv

    env = MiniHackBoxobanUnfilteredEnv()
    env.reset(seed=0)
    env._box_positions[0] = env._target_positions[0]

    _, reward, terminated, _, info = env.step(_a(env, "WAIT"))

    assert not terminated
    assert reward == pytest.approx(1 / env._num_boxes)
    assert info["boxes_on_target"] >= 1


def test_corridor_key_pickup_progress_reward_is_one_shot() -> None:
    from glyphbench.envs.minihack.corridor import MiniHackCorridorR5Env
    from glyphbench.envs.minihack.items import BRASS_KEY

    env = MiniHackCorridorR5Env()
    env.reset(seed=0)
    env._inventory.append(BRASS_KEY)

    _, reward1, _, _, info1 = env.step(_a(env, "WAIT"))
    _, reward2, _, _, info2 = env.step(_a(env, "WAIT"))

    assert reward1 == pytest.approx(env._PROGRESS_BUDGET / (env._num_rooms + 1))
    assert reward2 == 0.0
    assert "key" in info1["progress_milestones"]
    assert info2["progress_milestones"].count("key") == 1


def test_hidenseek_reaching_stairs_wins_before_adjacent_kobold_attack() -> None:
    from glyphbench.envs.minihack.hidenseek import MiniHackHideNSeekMappedEnv

    env = MiniHackHideNSeekMappedEnv()
    env.reset(seed=0)
    env._player_pos = (9, 9)
    env._player_hp = 2
    env._creatures[0].x, env._creatures[0].y = (9, 10)

    _, reward, terminated, _, info = env.step(_a(env, "MOVE_SE"))

    assert terminated
    assert reward == 1.0
    assert info["goal_reached"] is True


def test_memento_f2_wrong_floor2_submission_terminates_without_reward() -> None:
    from glyphbench.envs.minihack.memento import MiniHackMementoF2Env

    env = MiniHackMementoF2Env()
    env.reset(seed=0)
    env._floor = 2
    env._marker_pos = (3, 3)
    env._generate_floor()
    env._player_pos = (1, 3)

    _, move_reward, move_done, _, _ = env.step(_a(env, "MOVE_E"))
    _, reward, terminated, _, info = env.step(_a(env, "WAIT"))

    assert not move_done
    assert move_reward == 0.0
    assert terminated
    assert reward == 0.0
    assert info["goal_reached"] is False


def test_wield_unarmed_bump_still_allows_monster_turn() -> None:
    from glyphbench.envs.minihack.creatures import KOBOLD, Creature
    from glyphbench.envs.minihack.skill_wield import MiniHackWieldEnv

    env = MiniHackWieldEnv()
    env.reset(seed=0)
    env._player_pos = (3, 3)
    env._player_hp = 12
    env._creatures = [Creature.spawn(KOBOLD, 4, 3)]

    _, reward, terminated, _, _ = env.step(_a(env, "MOVE_E"))

    assert not terminated
    assert reward == 0.0
    assert env._player_hp < 12


def test_wod_pro_unidentified_wand_does_not_leak_on_pickup() -> None:
    from glyphbench.envs.minihack.skill_wod import MiniHackWoDProEnv

    env = MiniHackWoDProEnv()
    env.reset(seed=0)
    wand_pos = next(pos for pos, items in env._floor_items.items() if items[0].item_type == "wand")
    env._player_pos = wand_pos

    obs, _, _, _, _ = env.step(_a(env, "PICKUP"))

    assert "unidentified wand" in obs
    assert "wand of death" not in obs
    assert "wand of fire" not in obs
    assert "wand of cold" not in obs


def test_climber_star_reward_survives_enemy_collision() -> None:
    from glyphbench.envs.procgen.climber import ClimberEnv

    env = ClimberEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = (5, 10)
    env._set_cell(6, 10, "*")
    env._total_stars = 1
    env._entities = []
    env._add_entity("enemy", "E", 6, 10, dx=0)

    _, reward, terminated, _, _ = env.step(_a(env, "RIGHT"))

    assert terminated
    assert reward == pytest.approx(env._STAR_BUDGET / env._STAR_TARGET)


def test_fruitbot_negative_budgets_stay_within_lower_bound() -> None:
    from glyphbench.envs.procgen.fruitbot import FruitBotEnv

    assert FruitBotEnv._OBSTACLE_BUDGET + abs(FruitBotEnv._WALL_BOUNCE_BUDGET) <= 1.0


def test_heist_key_pickup_gives_partial_reward() -> None:
    from glyphbench.envs.procgen.heist import HeistEnv

    env = HeistEnv()
    env.reset(seed=0)
    env._set_cell(env._agent_x + 1, env._agent_y, "r")

    _, reward, terminated, _, info = env.step(_a(env, "RIGHT"))

    assert not terminated
    assert reward == pytest.approx(env._KEY_BUDGET / 3)
    assert "r" in info["keys_held"]


def test_heist_same_step_key_and_goal_is_additive() -> None:
    from glyphbench.envs.procgen.heist import HeistEnv

    env = HeistEnv()
    env.reset(seed=0)
    env._goal_x = env._agent_x + 1
    env._goal_y = env._agent_y
    env._set_cell(env._goal_x, env._goal_y, "r")

    _, reward, terminated, _, info = env.step(_a(env, "RIGHT"))

    assert terminated
    assert reward == pytest.approx(env._GOAL_REWARD + env._KEY_BUDGET / 3)
    assert "r" in info["keys_held"]


def test_snake_food_targets_emit_meaningful_rewards() -> None:
    from glyphbench.envs.classics.snake import (
        SnakeEasyEnv,
        SnakeHardEnv,
        SnakeMediumEnv,
    )

    assert SnakeEasyEnv()._target_food == 5
    assert SnakeMediumEnv()._target_food == 10
    assert SnakeHardEnv()._target_food == 20


def test_procgen_count_targets_emit_meaningful_rewards() -> None:
    from glyphbench.envs.procgen.bigfish import BigFishEnv
    from glyphbench.envs.procgen.bossfight import _BOSS_MAX_HP, BossFightEnv
    from glyphbench.envs.procgen.caveflyer import CaveFlyerEnv
    from glyphbench.envs.procgen.dodgeball import DodgeballEnv
    from glyphbench.envs.procgen.plunder import PlunderEnv
    from glyphbench.envs.procgen.starpilot import StarPilotEnv

    assert BigFishEnv._WIN_TARGET == 10
    assert DodgeballEnv._WIN_TARGET == 10
    assert PlunderEnv._WIN_TARGET == 10
    # StarPilot 2026-05 retune: kill quota raised to 12 (was 8) so a stationary
    # FIRE-spammer can no longer reach the cap from its single lane. Per-kill
    # reward (0.8/12 ~= 0.067) stays comfortably meaningful (>= 0.05).
    assert StarPilotEnv._ENEMY_TARGET == 12
    assert StarPilotEnv._ENEMY_BUDGET / StarPilotEnv._ENEMY_TARGET >= 0.05
    assert pytest.approx(0.1) == StarPilotEnv._POWERUP_BUDGET / StarPilotEnv._POWERUP_TARGET
    assert pytest.approx(0.1) == BossFightEnv._HIT_BUDGET / _BOSS_MAX_HP

    caveflyer = CaveFlyerEnv()
    caveflyer.reset(seed=0)
    assert caveflyer._total_enemies == 4
    assert caveflyer._ENEMY_BUDGET / caveflyer._total_enemies == pytest.approx(0.1)


def test_fruitbot_first_fruit_and_obstacle_are_meaningful() -> None:
    from glyphbench.envs.procgen.fruitbot import FruitBotEnv

    env = FruitBotEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = (5, 4)
    env._set_cell(5, 5, "%")

    _, fruit_reward, _, _, _ = env.step(_a(env, "NOOP"))

    assert fruit_reward == pytest.approx(0.1)

    env.reset(seed=0)
    env._agent_x, env._agent_y = (5, 4)
    env._set_cell(5, 5, "x")

    _, obstacle_reward, _, _, _ = env.step(_a(env, "NOOP"))

    assert obstacle_reward == pytest.approx(-0.1)


def test_chaser_first_pellet_is_meaningful() -> None:
    from glyphbench.envs.procgen.chaser import _PELLET, ChaserEnv

    env = ChaserEnv()
    env.reset(seed=0)
    env._entities = []
    env._set_cell(env._agent_x + 1, env._agent_y, _PELLET)

    _, reward, _, _, _ = env.step(_a(env, "RIGHT"))

    # Pellet reward is the +0.5 budget shared evenly across the full maze
    # (retune 2026-05: distributed over _initial_pellet_count instead of a
    # fixed milestone of 5, so a trivial walker cannot saturate the signal
    # by eating a handful of pellets). Each pickup yields a positive reward
    # that, summed over a full clear, equals the +0.5 pellet budget.
    assert reward > 0.0
    assert reward == pytest.approx(0.5 / env._initial_pellet_count)


def test_climber_and_miner_progress_rewards_are_meaningful() -> None:
    from glyphbench.envs.procgen.climber import ClimberEnv
    from glyphbench.envs.procgen.miner import MinerEnv

    climber = ClimberEnv()
    climber.reset(seed=0)
    climber._entities = []
    climber._agent_x, climber._agent_y = (5, 5)
    climber._set_cell(6, 5, "*")

    _, star_reward, _, _, _ = climber.step(_a(climber, "RIGHT"))

    assert star_reward == pytest.approx(0.1)

    miner = MinerEnv()
    miner.reset(seed=0)
    miner._set_cell(miner._agent_x + 1, miner._agent_y, "D")

    _, diamond_reward, _, _, _ = miner.step(_a(miner, "RIGHT"))

    assert diamond_reward == pytest.approx(0.1)


def test_jumper_gap_bottom_is_terminal() -> None:
    from glyphbench.envs.procgen.jumper import JumperEnv

    env = JumperEnv()
    env.reset(seed=0)
    env._agent_x = 5
    env._agent_y = env._world_h - 2
    env._set_cell(env._agent_x, env._world_h - 2, "·")
    env._set_cell(env._agent_x, env._world_h - 1, "·")
    env._on_ground = False
    env._jump_step = -1

    _, reward, terminated, _, info = env.step(_a(env, "NOOP"))

    assert terminated
    assert reward == 0.0
    assert info["killed_by"] == "fall"


def test_ninja_moving_shuriken_kill_gets_reward() -> None:
    from glyphbench.envs.procgen.ninja import NinjaEnv

    env = NinjaEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = (5, 9)
    env._facing = 1
    env._entities = []
    env._add_entity("enemy", "E", 7, 9, dx=0)
    env._total_enemies = 1
    env._enemies_killed = 0

    _, reward, terminated, _, _ = env.step(_a(env, "THROW"))

    assert not terminated
    # Kill yields the full enemy budget; the step also charges the small
    # per-step time cost, so the net step reward is budget - step cost.
    assert reward == pytest.approx(env._ENEMY_BUDGET - env._STEP_COST)
    assert env._enemies_killed == 1
