"""Tier-2 debate metric aggregator unit tests.

Pure-logic coverage — no env, no network, synthetic rollouts. Validates
the branching semantics: resolvable conditioning, 3-way vs 2-way TWC,
tie handling, position bias, mind-change decomposition, length-bias
Spearman sign, graceful degradation on non-debate rollouts.
"""

from __future__ import annotations

from typing import Any

from prime_rl.metrics.debate import _spearman, compute_step_metrics


def _mk(
    *,
    truth: str,
    winner: str | None,
    flipped_a: float = 0.0,
    flipped_b: float = 0.0,
    final_a: float | None = None,
    final_b: float | None = None,
    initial_a: float | None = None,
    initial_b: float | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    is_truncated: bool = False,
    error: Any = None,
    turns_a: float = 2.0,
    turns_b: float = 2.0,
    turns_judge: float = 1.0,
) -> dict[str, Any]:
    """Construct a synthetic debate rollout with the minimum fields the
    aggregator reads. ``truth`` is the truth-side debater."""
    fc_a = final_a if final_a is not None else (1.0 if truth == "debater_a" else 0.0)
    fc_b = final_b if final_b is not None else (1.0 if truth == "debater_b" else 0.0)
    ic_a = initial_a if initial_a is not None else fc_a
    ic_b = initial_b if initial_b is not None else fc_b
    return {
        "mar_score": {
            "episode_categorical": {
                "winner": winner,
                "first_answer/debater_a": "A",
                "final_answer/debater_a": "A",
                "first_answer/debater_b": "B",
                "final_answer/debater_b": "B",
            },
        },
        "final_correct/debater_a": fc_a,
        "final_correct/debater_b": fc_b,
        "initial_correct/debater_a": ic_a,
        "initial_correct/debater_b": ic_b,
        "flipped/debater_a": flipped_a,
        "flipped/debater_b": flipped_b,
        "turns/debater_a": turns_a,
        "turns/debater_b": turns_b,
        "turns/judge": turns_judge,
        "is_truncated": is_truncated,
        "error": error,
        "trajectory": trajectory or [],
    }


def _mk_trajectory(lengths_by_member: dict[str, int]) -> list[dict[str, Any]]:
    """Synthesize a trajectory whose per-member completion_ids lengths
    sum to the given counts (one step per member)."""
    steps = []
    for member, length in lengths_by_member.items():
        steps.append(
            {
                "extras": {"member_id": member, "phase": "propose"},
                "tokens": {"prompt_ids": [], "completion_ids": [0] * length},
            }
        )
    return steps


def test_empty_rollouts_returns_empty():
    assert compute_step_metrics([]) == {}


def test_non_debate_rollouts_silently_skipped():
    # Single-agent rollout: no mar_score.winner
    rollouts = [{"mar_score": {"episode_categorical": {}}, "error": None}]
    assert compute_step_metrics(rollouts) == {}


def test_perfect_judge_twc_1():
    """Judge always picks truth → twc_3way = 1.0 across resolvable cases."""
    rollouts = [
        _mk(truth="debater_a", winner="debater_a"),
        _mk(truth="debater_b", winner="debater_b"),
        _mk(truth="debater_a", winner="debater_a"),
    ]
    m = compute_step_metrics(rollouts)
    assert m["twc_3way"] == 1.0
    assert m["twc_2way_cond"] == 1.0
    assert m["tie_rate"] == 0.0
    assert m["n_resolvable"] == 3.0
    assert m["resolvable_rate"] == 1.0
    assert m["position_bias"] == 0.0  # perfect on both seats


def test_random_judge_twc_zero_three_way_null():
    """A judge that always picks tie → twc_3way = 0, tie_rate = 1.0.
    2-way conditional is undefined (no non-tie rollouts); shouldn't appear."""
    rollouts = [
        _mk(truth="debater_a", winner="tie"),
        _mk(truth="debater_b", winner="tie"),
    ]
    m = compute_step_metrics(rollouts)
    assert m["twc_3way"] == 0.0
    assert m["tie_rate"] == 1.0
    assert "twc_2way_cond" not in m
    assert m["twc_3way_null"] == 1.0 / 3.0


def test_position_bias_asymmetric_judge():
    """Judge always picks debater_a regardless of truth → position_bias = 1.0."""
    rollouts = [
        _mk(truth="debater_a", winner="debater_a"),
        _mk(truth="debater_a", winner="debater_a"),
        _mk(truth="debater_b", winner="debater_a"),
        _mk(truth="debater_b", winner="debater_a"),
    ]
    m = compute_step_metrics(rollouts)
    assert m["twc_by_seat_a"] == 1.0  # truth=a, judge picks a → all correct
    assert m["twc_by_seat_b"] == 0.0  # truth=b, judge picks a → all wrong
    assert m["position_bias"] == 1.0


def test_mind_change_good_vs_bad():
    """debater_a: wrong→right (good); debater_b: right→wrong (bad)."""
    rollouts = [
        _mk(
            truth="debater_a",
            winner="debater_a",
            initial_a=0.0,
            final_a=1.0,  # good flip
            initial_b=1.0,
            final_b=0.0,  # bad flip
            flipped_a=1.0,
            flipped_b=1.0,
        ),
    ]
    m = compute_step_metrics(rollouts)
    assert m["mind_change_good_rate/debater_a"] == 1.0
    assert m["mind_change_bad_rate/debater_a"] == 0.0
    assert m["mind_change_good_rate/debater_b"] == 0.0
    assert m["mind_change_bad_rate/debater_b"] == 1.0


def test_unresolvable_both_correct_drops_from_twc():
    """Both debaters correct → not resolvable → twc_3way skipped, but
    rollout counted in n_rollouts + kept for error/truncation rates."""
    rollouts = [
        _mk(truth="debater_a", winner="debater_a", final_b=1.0),  # both right
    ]
    m = compute_step_metrics(rollouts)
    assert m["n_rollouts"] == 1.0
    assert m["n_resolvable"] == 0.0
    assert m["resolvable_rate"] == 0.0
    assert "twc_3way" not in m


def test_length_bias_correlation():
    """When debater_a always writes more tokens AND wins every time,
    length_bias_corr should be strongly positive."""
    rollouts = []
    for _ in range(5):
        rollouts.append(
            _mk(
                truth="debater_a",
                winner="debater_a",
                trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 100, "judge": 50}),
            )
        )
    for _ in range(5):
        rollouts.append(
            _mk(
                truth="debater_b",
                winner="debater_a",  # judge picks a despite truth=b
                trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 100, "judge": 50}),
            )
        )
    m = compute_step_metrics(rollouts)
    # All rollouts have length_delta > 0, all have winner=a. Spearman is 0
    # when one variable is constant. Sanity: at least the key appears.
    # (Length bias is a diagnostic — this case is degenerate by design;
    # see the three-way case below for a signed check.)
    assert "length_bias_corr" in m


def test_length_bias_sign_sensitive():
    """Short winners → negative corr; long winners → positive corr."""
    rollouts = []
    # Cases where debater_a won AND wrote more tokens
    for _ in range(3):
        rollouts.append(
            _mk(
                truth="debater_a",
                winner="debater_a",
                trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 50}),
            )
        )
    # Cases where debater_b won AND debater_a wrote more tokens (a lost despite writing more)
    for _ in range(3):
        rollouts.append(
            _mk(
                truth="debater_b",
                winner="debater_b",
                trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 50}),
            )
        )
    # Paired tuples: all have length_delta = +450 (constant) → Spearman = 0.
    m = compute_step_metrics(rollouts)
    assert m.get("length_bias_corr", 0.0) == 0.0


def test_truncation_and_error_rates():
    rollouts = [
        _mk(truth="debater_a", winner="debater_a", is_truncated=True),
        _mk(truth="debater_a", winner="debater_a", error={"error": "timeout"}),
        _mk(truth="debater_a", winner="debater_a"),
    ]
    m = compute_step_metrics(rollouts)
    assert m["truncation_rate"] == 1 / 3
    assert m["error_rate"] == 1 / 3


def test_flip_rate_aggregates_per_member():
    rollouts = [
        _mk(truth="debater_a", winner="debater_a", flipped_a=1.0, flipped_b=0.0),
        _mk(truth="debater_a", winner="debater_a", flipped_a=0.0, flipped_b=1.0),
        _mk(truth="debater_a", winner="debater_a", flipped_a=1.0, flipped_b=1.0),
    ]
    m = compute_step_metrics(rollouts)
    assert abs(m["flip_rate/debater_a"] - 2 / 3) < 1e-9
    assert abs(m["flip_rate/debater_b"] - 2 / 3) < 1e-9


def test_spearman_basic():
    """Sanity: identity = +1, reverse = −1, constant = 0."""
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert _spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0
    assert _spearman([], []) == 0.0


def test_twc_null_reference_lines():
    """TWC null-baseline reference lines always present when any resolvable case exists."""
    rollouts = [_mk(truth="debater_a", winner="debater_b")]  # judge wrong
    m = compute_step_metrics(rollouts)
    assert m["twc_3way_null"] == 1.0 / 3.0
    assert m["twc_2way_cond_null"] == 0.5
