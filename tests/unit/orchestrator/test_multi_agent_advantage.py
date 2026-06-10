"""Rank-7 RAE acceptance tests: shrunk leave-one-out + historical prior,
order-invariant, antithetic merge for zero-sum member pairs.

Frame derivation and zero-sum validation operate on the UNFILTERED
``episode_pairs`` (mar_score view), never on the trainability-filtered
member rows — under train_one a whole group can collapse to one seat.
"""

import random

import pytest
from verifiers.types import MemberRollout

from prime_rl.configs.multi_agent import MultiAgentConfig
from prime_rl.configs.orchestrator import DefaultAdvantageConfig, RAEAdvantageConfig
from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    extract_episode_pairs_for_multi_agent,
    fan_out_for_multi_agent,
    validate_advantage_mode,
)

ENV_NAME = "debate_v1"
KEY = (ENV_NAME, 1)


def _member(
    *,
    example_id: int | str = 1,
    member_id: str = "debater_a",
    reward: float = 1.0,
    episode_id: str = "ep-0",
    task: object = ENV_NAME,
    env_name: str | None = None,
) -> MemberRollout:
    rollout = MemberRollout(
        example_id=example_id,
        task=task,
        trajectory=[],
        sampling_args={"temperature": 0.7},
        error=None,
        reward=reward,
        episode_id=episode_id,
        member_id=member_id,
    )
    if env_name is not None:
        rollout["env_name"] = env_name
    return rollout


def _pair_rows(
    episode_id: str, reward: float, *, example_id: int | str = 1, env_name: str | None = None
) -> list[MemberRollout]:
    """Both rows of one zero-sum episode: canonical ``debater_a`` gets ``reward``."""
    return [
        _member(member_id="debater_a", reward=reward, episode_id=episode_id, example_id=example_id, env_name=env_name),
        _member(member_id="debater_b", reward=-reward, episode_id=episode_id, example_id=example_id, env_name=env_name),
    ]


def _group(canonical_rewards: list[float]) -> list[MemberRollout]:
    rows: list[MemberRollout] = []
    for i, reward in enumerate(canonical_rewards):
        rows.extend(_pair_rows(f"ep-{i}", reward))
    return rows


def _pairs(canonical_rewards: list[float]) -> dict[str, dict[str, float]]:
    """Full (unfiltered) zero-sum pairs, one per episode."""
    return {f"ep-{i}": {"debater_a": r, "debater_b": -r} for i, r in enumerate(canonical_rewards)}


def test_permutation_equivariance():
    """Advantages and post-call baselines are invariant under any reordering
    of the member rows — exact comparison, no fp tolerance."""
    canonical_rewards = [1.0, -1.0, 0.0, 1.0]
    rows = _group(canonical_rewards)
    episode_pairs = _pairs(canonical_rewards)

    def run(ordered_rows: list[MemberRollout]) -> tuple[dict[tuple[str, str], float], dict, list]:
        state = RAEState(baselines={KEY: 0.25})
        advantages, _ = compute_rae_advantages(ordered_rows, state, episode_pairs=episode_pairs)
        by_identity = {(r["episode_id"], r["member_id"]): a for r, a in zip(ordered_rows, advantages)}
        return by_identity, dict(state.baselines), state.last_folds

    reference, reference_baselines, _ = run(rows)
    for seed in range(5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        result, baselines, folds = run(shuffled)
        assert result == reference
        assert baselines == reference_baselines
        assert [(f.baseline_before, f.baseline_after, f.group_size, f.canonical) for f in folds] == [
            (0.25, reference_baselines[KEY], 4, "debater_a")
        ]


def test_antithetic_invariant():
    canonical_rewards = [1.0, -1.0, 0.0, 1.0]
    rows = _group(canonical_rewards)
    state = RAEState(baselines={KEY: 0.25})

    advantages, _ = compute_rae_advantages(rows, state, episode_pairs=_pairs(canonical_rewards))

    # Per pair: A_b is the exact negation of A_a
    for i in range(0, len(rows), 2):
        assert advantages[i] + advantages[i + 1] == 0.0
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)


def test_g1_reduces_to_pure_prior():
    state = RAEState(baselines={KEY: 0.4})

    advantages, _ = compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))

    # G=1: lam pinned to 1, LOO term has zero weight. A = r - b = 1.0 - 0.4 = 0.6
    assert advantages == [pytest.approx(0.6), pytest.approx(-0.6)]


def test_g1_with_neff_zero_still_uses_prior():
    state = RAEState(baselines={KEY: 0.4}, n_eff=0.0)

    advantages, _ = compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))

    # lam = 1 at G=1 regardless of n_eff (0/0 would otherwise be NaN)
    assert advantages == [pytest.approx(0.6), pytest.approx(-0.6)]


def test_neff_zero_reduces_to_rloo():
    state = RAEState(baselines={KEY: 0.7}, n_eff=0.0)

    advantages, _ = compute_rae_advantages(_group([1.0, -1.0, 1.0]), state, episode_pairs=_pairs([1.0, -1.0, 1.0]))

    # lam = 0: A_i = r_i - mean(r_{-i}); the 0.7 prior carries zero weight.
    #   A_0 = 1 - (-1+1)/2 = 1.0;  A_1 = -1 - (1+1)/2 = -2.0;  A_2 = 1 - (1-1)/2 = 1.0
    assert advantages == [1.0, -1.0, -2.0, 2.0, 1.0, -1.0]


def test_fold_semantics_and_mid_group_isolation():
    state = RAEState(baselines={KEY: 0.5})

    advantages, _ = compute_rae_advantages(_group([1.0, 1.0]), state, episode_pairs=_pairs([1.0, 1.0]))

    # Both episodes subtract the same pre-group b0 = 0.5 — identical advantages
    # prove the fold is not consulted mid-group. lam = 6/(6+1) = 6/7:
    #   A = 1 - (6/7 * 0.5 + 1/7 * 1.0) = 1 - 4/7 = 3/7 = 0.42857142857142855
    assert advantages[0] == advantages[2]
    assert advantages[0] == pytest.approx(0.42857142857142855)
    # Fold once at group close: b1 = beta*b0 + (1-beta)*mean = 0.9*0.5 + 0.1*1.0 = 0.55
    assert state.baselines[KEY] == pytest.approx(0.55)
    (fold,) = state.last_folds
    assert (fold.key, fold.group_size, fold.cold, fold.canonical) == (KEY, 2, False, "debater_a")
    assert fold.baseline_before == 0.5
    assert fold.baseline_after == pytest.approx(0.55)
    assert fold.group_mean == pytest.approx(1.0)


def test_cold_key_baseline_is_zero():
    state = RAEState()

    advantages, _ = compute_rae_advantages(_group([1.0, -1.0]), state, episode_pairs=_pairs([1.0, -1.0]))

    # Cold key: b0 = 0. lam = 6/7:
    #   A_0 = 1 - (0 + 1/7 * (-1)) = 8/7 = 1.1428571428571428
    #   A_1 = -1 - (0 + 1/7 * 1) = -8/7
    assert advantages[0] == pytest.approx(1.1428571428571428)
    assert advantages[2] == pytest.approx(-1.1428571428571428)
    # Fold: b1 = 0.9*0 + 0.1*mean([1,-1]) = 0.0 — but the key is now warm
    assert state.baselines[KEY] == pytest.approx(0.0)
    assert state.canonical_members[KEY] == "debater_a"
    assert state.last_folds[0].cold is True


def test_zero_sum_violation_raises():
    state = RAEState()
    rows = _pair_rows("ep-0", 1.0)

    with pytest.raises(ValueError, match="zero-sum member pairs") as excinfo:
        compute_rae_advantages(rows, state, episode_pairs={"ep-0": {"debater_a": 1.0, "debater_b": 1.0}})
    assert "ep-0" in str(excinfo.value)
    assert "debater_a=1.0" in str(excinfo.value)


def test_non_zero_sum_single_seat_episode_raises():
    """train_one shape: only one seat survives filtering, but validation
    runs on the full pair from mar_score — the violation cannot hide."""
    state = RAEState()
    rows = [_member(member_id="debater_a", reward=1.0, episode_id="ep-0")]

    with pytest.raises(ValueError, match="zero-sum member pairs"):
        compute_rae_advantages(rows, state, episode_pairs={"ep-0": {"debater_a": 1.0, "debater_b": 0.0}})


def test_more_than_two_pair_members_raise():
    state = RAEState()
    rows = _pair_rows("ep-0", 1.0)
    episode_pairs = {"ep-0": {"debater_a": 1.0, "debater_b": -1.0, "judge": 0.0}}

    with pytest.raises(ValueError, match="zero-sum member pairs"):
        compute_rae_advantages(rows, state, episode_pairs=episode_pairs)


def test_duplicate_member_row_in_episode_raises():
    state = RAEState()
    rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
    ]

    with pytest.raises(ValueError, match="duplicate member rows"):
        compute_rae_advantages(rows, state, episode_pairs=_pairs([1.0]))


def test_member_row_outside_pair_raises():
    state = RAEState()
    rows = [_member(member_id="judge", reward=0.0, episode_id="ep-0")]

    with pytest.raises(ValueError, match="not in the episode's zero-sum pair"):
        compute_rae_advantages(rows, state, episode_pairs=_pairs([1.0]))


def test_quarantined_single_member_episode_uses_sign():
    state = RAEState()
    # ep-1's mar_score genuinely lacks debater_a; canonical reward = -r_b = -1.0
    rows = _pair_rows("ep-0", 1.0) + [_member(member_id="debater_b", reward=1.0, episode_id="ep-1")]
    episode_pairs = {"ep-0": {"debater_a": 1.0, "debater_b": -1.0}, "ep-1": {"debater_b": 1.0}}

    advantages, _ = compute_rae_advantages(rows, state, episode_pairs=episode_pairs)

    # Canonical rewards [1, -1], cold b0 = 0, lam = 6/7:
    #   A_canonical(ep-1) = -1 - 1/7 = -8/7; the present debater_b row gets +8/7
    assert advantages[2] == pytest.approx(1.1428571428571428)
    assert advantages[0] == pytest.approx(1.1428571428571428)
    assert state.canonical_members[KEY] == "debater_a"


def test_singleton_pair_group_inherits_persisted_frame():
    """A group of quarantined single-member episodes cannot establish a
    frame on its own — it sign-derives against the persisted canonical."""
    state = RAEState(baselines={KEY: 0.5}, canonical_members={KEY: "debater_a"})
    rows = [_member(member_id="debater_b", reward=1.0, episode_id="ep-0")]

    advantages, _ = compute_rae_advantages(rows, state, episode_pairs={"ep-0": {"debater_b": 1.0}})

    # Canonical reward = -1. G=1: A_canonical = -1 - 0.5 = -1.5; the b row gets +1.5
    assert advantages == [pytest.approx(1.5)]
    # Fold stays in the a-frame: b1 = 0.9*0.5 + 0.1*(-1) = 0.35
    assert state.baselines[KEY] == pytest.approx(0.35)
    assert state.canonical_members[KEY] == "debater_a"


def test_frame_change_fails_loud():
    state = RAEState(baselines={KEY: 0.5}, canonical_members={KEY: "debater_b"})

    with pytest.raises(ValueError, match="persisted baseline frame cannot flip"):
        compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))


def test_single_seat_group_only_debater_b_keeps_canonical_frame():
    """train_one collapse: every surviving row is debater_b, yet the frame
    comes from the full pairs — the baseline folds in the a-frame."""
    state = RAEState(baselines={KEY: 0.5})
    rows = [
        _member(member_id="debater_b", reward=-1.0, episode_id="ep-0"),
        _member(member_id="debater_b", reward=1.0, episode_id="ep-1"),
    ]

    advantages, _ = compute_rae_advantages(rows, state, episode_pairs=_pairs([1.0, -1.0]))

    # Canonical rewards (a-frame) [1, -1], b0 = 0.5, lam = 6/7:
    #   A_canonical(ep-0) = 1 - (6/7*0.5 + 1/7*(-1)) = 1 - 2/7 = 5/7 -> b row gets -5/7
    #   A_canonical(ep-1) = -1 - (6/7*0.5 + 1/7*1) = -1 - 4/7 = -11/7 -> b row gets +11/7
    assert advantages[0] == pytest.approx(-0.7142857142857143)
    assert advantages[1] == pytest.approx(1.5714285714285714)
    # Fold in the a-frame: b1 = 0.9*0.5 + 0.1*mean([1,-1]) = 0.45
    assert state.baselines[KEY] == pytest.approx(0.45)
    assert state.canonical_members[KEY] == "debater_a"


def test_cross_step_frame_stability_under_seat_collapse():
    """An all-debater_b group followed by an all-debater_a group on the same
    key: the persisted baseline must stay in one frame throughout."""
    state = RAEState()

    # Step 1: train_one kept only debater_b; debater_a won both episodes
    step1_rows = [
        _member(member_id="debater_b", reward=-1.0, episode_id="ep-0"),
        _member(member_id="debater_b", reward=-1.0, episode_id="ep-1"),
    ]
    step1, _ = compute_rae_advantages(step1_rows, state, episode_pairs=_pairs([1.0, 1.0]))
    # Cold b0 = 0, canonical rewards [1, 1], lam = 6/7:
    #   A_canonical = 1 - (0 + 1/7*1) = 6/7 -> b rows get -6/7 = -0.8571428571428571
    assert step1 == [pytest.approx(-0.8571428571428571), pytest.approx(-0.8571428571428571)]
    # Fold in the a-frame: b = 0.9*0 + 0.1*1 = 0.1 (a is winning -> baseline rises)
    assert state.baselines[KEY] == pytest.approx(0.1)
    assert state.canonical_members[KEY] == "debater_a"

    # Step 2: train_one kept only debater_a; debater_a won both episodes again
    step2_rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_a", reward=1.0, episode_id="ep-1"),
    ]
    step2, _ = compute_rae_advantages(step2_rows, state, episode_pairs=_pairs([1.0, 1.0]))
    # b0 = 0.1 consumed in the SAME a-frame:
    #   A = 1 - (6/7*0.1 + 1/7*1) = 1 - 1.6/7 = 27/35 = 0.7714285714285714
    assert step2 == [pytest.approx(0.7714285714285714), pytest.approx(0.7714285714285714)]
    # Fold: b = 0.9*0.1 + 0.1*1 = 0.19
    assert state.baselines[KEY] == pytest.approx(0.19)
    assert state.canonical_members[KEY] == "debater_a"


def test_single_seat_group_trains_on_canonical_rewards():
    """train_one collapse onto the canonical seat itself."""
    state = RAEState(baselines={KEY: 0.5})
    rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_a", reward=-1.0, episode_id="ep-1"),
    ]

    advantages, _ = compute_rae_advantages(rows, state, episode_pairs=_pairs([1.0, -1.0]))

    # lam = 6/7:
    #   A_0 = 1 - (6/7*0.5 + 1/7*(-1)) = 1 - 2/7 = 5/7 = 0.7142857142857143
    #   A_1 = -1 - (6/7*0.5 + 1/7*1) = -1 - 4/7 = -11/7 = -1.5714285714285714
    assert advantages[0] == pytest.approx(0.7142857142857143)
    assert advantages[1] == pytest.approx(-1.5714285714285714)
    # Fold: b1 = 0.9*0.5 + 0.1*0 = 0.45
    assert state.baselines[KEY] == pytest.approx(0.45)


def test_baselines_partition_by_env_and_example():
    state = RAEState()
    rows = (
        _pair_rows("ep-0", 1.0, env_name="env_a", example_id=1)
        + _pair_rows("ep-1", -1.0, env_name="env_a", example_id=2)
        + _pair_rows("ep-2", 1.0, env_name="env_b", example_id=1)
    )
    episode_pairs = {
        "ep-0": {"debater_a": 1.0, "debater_b": -1.0},
        "ep-1": {"debater_a": -1.0, "debater_b": 1.0},
        "ep-2": {"debater_a": 1.0, "debater_b": -1.0},
    }

    compute_rae_advantages(rows, state, episode_pairs=episode_pairs)

    # Each (env_name, example_id) is its own G=1 group; fold = 0.1 * r
    assert state.baselines == {
        ("env_a", 1): pytest.approx(0.1),
        ("env_a", 2): pytest.approx(-0.1),
        ("env_b", 1): pytest.approx(0.1),
    }
    assert len(state.last_folds) == 3


def test_fan_out_carries_env_name_for_rae_identity():
    state = RAEState()
    episode = {
        "env_name": "gpqa_debate",
        "example_id": 1,
        "task": {"question": "q"},
        "trajectory": [{"extras": {"member_id": m}} for m in ("prover", "verifier")],
        "sampling_args": {"temperature": 0.7},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [
                {"member_id": "prover", "reward": 1.0},
                {"member_id": "verifier", "reward": -1.0},
            ],
            "episode_scalar": 1.0,
        },
    }

    units, mapping = fan_out_for_multi_agent([episode])
    episode_pairs = extract_episode_pairs_for_multi_agent([episode], MultiAgentConfig())
    compute_rae_advantages(units, state, episode_pairs=episode_pairs)

    assert mapping == [[0, 1]]
    assert episode_pairs == {"ep-0": {"prover": 1.0, "verifier": -1.0}}
    assert set(state.baselines) == {("gpqa_debate", 1)}


def test_fan_out_respects_trainable_member_predicate():
    episode = {
        "env_name": ENV_NAME,
        "example_id": 1,
        "task": ENV_NAME,
        "trajectory": [{"extras": {"member_id": m}} for m in ("debater_a", "debater_b", "judge")],
        "sampling_args": {"temperature": 0.7},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [
                {"member_id": "debater_a", "reward": 1.0},
                {"member_id": "debater_b", "reward": -1.0},
                {"member_id": "judge", "reward": 0.0},
            ],
            "episode_scalar": 1.0,
        },
    }

    units, mapping = fan_out_for_multi_agent(
        [episode],
        is_trainable_member=lambda _rollout, member_id: member_id == "debater_a",
    )

    assert mapping == [[0]]
    assert [unit["member_id"] for unit in units] == ["debater_a"]


def test_rae_stats_track_cold_keys_drift_and_baselines():
    state = RAEState()

    # Cold key, one episode (G=1 => lambda=1, pure prior): A = +/-1.0,
    # fold b: 0 -> 0.9*0 + 0.1*1.0 = 0.1. Framed baselines: +0.1 / -0.1.
    _, stats = compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))

    assert stats.updates == 2
    assert stats.cold_updates == 1
    assert stats.baseline_keys_total == 1
    assert stats.baseline_abs_delta_sum == pytest.approx(0.1)
    assert stats.baseline_sum_by_member == {"debater_a": pytest.approx(0.1), "debater_b": pytest.approx(-0.1)}
    assert stats.updates_by_member == {"debater_a": 1, "debater_b": 1}

    # Warm key: fold 0.1 -> 0.9*0.1 + 0.1*1.0 = 0.19, delta 0.09.
    _, stats = compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))

    assert stats.cold_updates == 0
    assert stats.baseline_abs_delta_sum == pytest.approx(0.09)
    assert stats.baseline_keys_total == 1


def test_rae_stats_merge_accumulates_and_keeps_freshest_key_count():
    state = RAEState()
    _, merged = compute_rae_advantages(_pair_rows("ep-0", 1.0), state, episode_pairs=_pairs([1.0]))
    second_rows = _pair_rows("ep-0", 1.0, example_id=1) + _pair_rows("ep-1", 1.0, example_id=2)
    _, second = compute_rae_advantages(second_rows, state, episode_pairs=_pairs([1.0, 1.0]))

    merged.merge(second)

    assert merged.updates == 6
    assert merged.cold_updates == 2  # key (env, 1) cold once, key (env, 2) cold once
    assert merged.updates_by_member == {"debater_a": 3, "debater_b": 3}
    assert merged.baseline_keys_total == 2


def test_rae_requires_string_identity_when_task_is_the_fallback():
    state = RAEState()
    rollout = _member(task={"question": "q"}, env_name=None)

    with pytest.raises(TypeError, match="string env_name"):
        compute_rae_advantages([rollout], state, episode_pairs=_pairs([1.0]))


def test_validate_advantage_mode_rejects_rae_on_single_agent_env():
    with pytest.raises(ValueError, match="not a multi-agent env"):
        validate_advantage_mode("math", is_multi_agent=False, advantage=RAEAdvantageConfig())


def test_validate_advantage_mode_rejects_non_rae_on_multi_agent_env():
    with pytest.raises(ValueError, match="requires advantage.type='rae'"):
        validate_advantage_mode("debate", is_multi_agent=True, advantage=DefaultAdvantageConfig())
    with pytest.raises(ValueError, match="requires advantage.type='rae'"):
        validate_advantage_mode("debate", is_multi_agent=True, advantage=None)


def test_validate_advantage_mode_accepts_matched_pairings():
    validate_advantage_mode("debate", is_multi_agent=True, advantage=RAEAdvantageConfig())
    validate_advantage_mode("math", is_multi_agent=False, advantage=DefaultAdvantageConfig())
    validate_advantage_mode("math", is_multi_agent=False, advantage=None)
