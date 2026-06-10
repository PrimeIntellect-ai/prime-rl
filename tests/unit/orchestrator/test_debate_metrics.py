from __future__ import annotations

import json
from typing import Any

from prime_rl.metrics.debate import _spearman, compute_step_metrics, write_step_metrics
from prime_rl.utils.monitor.base import NoOpMonitor


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
    return [
        {
            "extras": {"member_id": member, "phase": "propose"},
            "tokens": {"prompt_ids": [], "completion_ids": [0] * length},
        }
        for member, length in lengths_by_member.items()
    ]


def test_empty_rollouts_returns_empty():
    assert compute_step_metrics([]) == {}


def test_non_debate_rollouts_are_skipped():
    assert compute_step_metrics([{"mar_score": {"episode_categorical": {}}, "error": None}]) == {}


def test_perfect_judge_twc_1():
    rollouts = [
        _mk(truth="debater_a", winner="debater_a"),
        _mk(truth="debater_b", winner="debater_b"),
        _mk(truth="debater_a", winner="debater_a"),
    ]
    metrics = compute_step_metrics(rollouts)
    assert metrics["twc_3way"] == 1.0
    assert metrics["twc_2way_cond"] == 1.0
    assert metrics["tie_rate"] == 0.0
    assert metrics["n_resolvable"] == 3.0
    assert metrics["resolvable_rate"] == 1.0
    assert metrics["position_bias"] == 0.0


def test_tie_judge_sets_three_way_null_without_two_way_metric():
    metrics = compute_step_metrics(
        [
            _mk(truth="debater_a", winner="tie"),
            _mk(truth="debater_b", winner="tie"),
        ]
    )
    assert metrics["twc_3way"] == 0.0
    assert metrics["tie_rate"] == 1.0
    assert "twc_2way_cond" not in metrics
    assert metrics["twc_3way_null"] == 1.0 / 3.0


def test_position_bias_asymmetric_judge():
    metrics = compute_step_metrics(
        [
            _mk(truth="debater_a", winner="debater_a"),
            _mk(truth="debater_a", winner="debater_a"),
            _mk(truth="debater_b", winner="debater_a"),
            _mk(truth="debater_b", winner="debater_a"),
        ]
    )
    assert metrics["twc_by_seat_a"] == 1.0
    assert metrics["twc_by_seat_b"] == 0.0
    assert metrics["position_bias"] == 1.0


def test_mind_change_good_vs_bad():
    metrics = compute_step_metrics(
        [
            _mk(
                truth="debater_a",
                winner="debater_a",
                initial_a=0.0,
                final_a=1.0,
                initial_b=1.0,
                final_b=0.0,
                flipped_a=1.0,
                flipped_b=1.0,
            ),
        ]
    )
    assert metrics["mind_change_good_rate/debater_a"] == 1.0
    assert metrics["mind_change_bad_rate/debater_a"] == 0.0
    assert metrics["mind_change_good_rate/debater_b"] == 0.0
    assert metrics["mind_change_bad_rate/debater_b"] == 1.0


def test_unresolvable_both_correct_drops_from_twc():
    metrics = compute_step_metrics([_mk(truth="debater_a", winner="debater_a", final_b=1.0)])
    assert metrics["n_rollouts"] == 1.0
    assert metrics["n_resolvable"] == 0.0
    assert metrics["resolvable_rate"] == 0.0
    assert "twc_3way" not in metrics


def test_length_bias_correlation_key_for_paired_decisions():
    rollouts = [
        _mk(
            truth="debater_a",
            winner="debater_a",
            trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 100, "judge": 50}),
        )
        for _ in range(5)
    ]
    rollouts.extend(
        _mk(
            truth="debater_b",
            winner="debater_a",
            trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 100, "judge": 50}),
        )
        for _ in range(5)
    )
    assert "length_bias_corr" in compute_step_metrics(rollouts)


def test_length_bias_returns_zero_for_constant_delta():
    rollouts = [
        _mk(
            truth="debater_a",
            winner="debater_a",
            trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 50}),
        )
        for _ in range(3)
    ]
    rollouts.extend(
        _mk(
            truth="debater_b",
            winner="debater_b",
            trajectory=_mk_trajectory({"debater_a": 500, "debater_b": 50}),
        )
        for _ in range(3)
    )
    assert compute_step_metrics(rollouts).get("length_bias_corr", 0.0) == 0.0


def test_truncation_and_error_rates():
    metrics = compute_step_metrics(
        [
            _mk(truth="debater_a", winner="debater_a", is_truncated=True),
            _mk(truth="debater_a", winner="debater_a", error={"error": "timeout"}),
            _mk(truth="debater_a", winner="debater_a"),
        ]
    )
    assert metrics["truncation_rate"] == 1 / 3
    assert metrics["error_rate"] == 1 / 3


def test_flip_rate_aggregates_per_member():
    metrics = compute_step_metrics(
        [
            _mk(truth="debater_a", winner="debater_a", flipped_a=1.0, flipped_b=0.0),
            _mk(truth="debater_a", winner="debater_a", flipped_a=0.0, flipped_b=1.0),
            _mk(truth="debater_a", winner="debater_a", flipped_a=1.0, flipped_b=1.0),
        ]
    )
    assert abs(metrics["flip_rate/debater_a"] - 2 / 3) < 1e-9
    assert abs(metrics["flip_rate/debater_b"] - 2 / 3) < 1e-9


def test_spearman_basic():
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert _spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0
    assert _spearman([], []) == 0.0


def test_twc_null_reference_lines():
    metrics = compute_step_metrics([_mk(truth="debater_a", winner="debater_b")])
    assert metrics["twc_3way_null"] == 1.0 / 3.0
    assert metrics["twc_2way_cond_null"] == 0.5


def _mk_pcd4_final(*, final_a: float, winner: str | None) -> dict[str, Any]:
    """Asymmetric pack rollout: debater_b is an answerless critic, so only
    debater_a carries final_correct, and the rubric emits the role-aware
    episode keys without the legacy all-debater aliases."""
    return {
        "mar_score": {"episode_categorical": {"winner": winner}},
        "final_correct/debater_a": final_a,
        "initial_correct/debater_a": final_a,
        "accuracy/debater_a": final_a,
        "any_answer_member_correct": final_a,
        "all_answer_members_correct": final_a,
        "flipped/debater_a": 0.0,
        "flipped/debater_b": 0.0,
        "turns/debater_a": 3.0,
        "turns/debater_b": 3.0,
        "turns/judge": 1.0,
        "is_truncated": False,
        "error": None,
        "trajectory": [],
    }


def test_answerless_critic_pack_resolves_truth_from_single_answer_member():
    metrics = compute_step_metrics(
        [
            _mk_pcd4_final(final_a=1.0, winner="debater_a"),  # truth=a, judge right
            _mk_pcd4_final(final_a=0.0, winner="debater_b"),  # truth=b, judge right
            _mk_pcd4_final(final_a=1.0, winner="debater_b"),  # truth=a, judge wrong
            _mk_pcd4_final(final_a=0.0, winner="debater_a"),  # truth=b, judge wrong
        ]
    )
    assert metrics["n_resolvable"] == 4.0
    assert metrics["resolvable_rate"] == 1.0
    assert metrics["twc_3way"] == 0.5
    assert metrics["twc_2way_cond"] == 0.5
    assert metrics["n_non_tie"] == 4.0
    assert metrics["tie_rate"] == 0.0
    assert metrics["twc_by_seat_a"] == 0.5
    assert metrics["twc_by_seat_b"] == 0.5
    assert metrics["position_bias"] == 0.0


def test_symmetric_pack_with_missing_member_key_stays_unresolvable():
    # all_debaters_correct present => every debater declares an answer, so a
    # missing final_correct key means extraction failure, not an answerless
    # seat. Single-sided truth inference would be wrong here.
    rollout = _mk(truth="debater_a", winner="debater_a")
    del rollout["final_correct/debater_b"]
    rollout["any_answer_member_correct"] = 1.0
    rollout["all_answer_members_correct"] = 0.0
    rollout["any_debater_correct"] = 1.0
    rollout["all_debaters_correct"] = 0.0
    metrics = compute_step_metrics([rollout])
    assert metrics["n_resolvable"] == 0.0


def test_single_member_key_without_episode_keys_stays_unresolvable():
    # No any_answer_member_correct => some answer-declaring member did not
    # resolve (grader error), so the pack shape is unknown.
    rollout = _mk(truth="debater_a", winner="debater_a")
    del rollout["final_correct/debater_b"]
    metrics = compute_step_metrics([rollout])
    assert metrics["n_resolvable"] == 0.0


def test_write_step_metrics_namespaces_payload_and_persists(tmp_path):
    monitor = NoOpMonitor()
    path = tmp_path / "step_7" / "train_debate_metrics.json"
    write_step_metrics(
        [_mk(truth="debater_a", winner="debater_a")],
        path=path,
        step=7,
        monitor=monitor,
        prefix="debate",
    )
    assert json.loads(path.read_text())["twc_3way"] == 1.0
    (payload,) = monitor.history
    assert payload["step"] == 7
    assert payload["debate/twc_3way"] == 1.0
    assert all(key == "step" or key.startswith("debate/") for key in payload)


def test_write_step_metrics_eval_prefix_matches_eval_namespace(tmp_path):
    monitor = NoOpMonitor()
    write_step_metrics(
        [_mk(truth="debater_a", winner="debater_b")],
        path=tmp_path / "eval_debate_metrics_gpqa_debate.json",
        step=3,
        monitor=monitor,
        prefix="eval/gpqa_debate/debate",
    )
    (payload,) = monitor.history
    assert payload["eval/gpqa_debate/debate/twc_3way"] == 0.0


def test_write_step_metrics_non_debate_batch_is_silent(tmp_path):
    monitor = NoOpMonitor()
    path = tmp_path / "train_debate_metrics.json"
    write_step_metrics(
        [{"mar_score": {"episode_categorical": {}}, "error": None}, {"reward": 1.0, "error": None}],
        path=path,
        step=1,
        monitor=monitor,
        prefix="debate",
    )
    assert not path.exists()
    assert monitor.history == []
