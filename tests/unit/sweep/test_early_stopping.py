from prime_rl.configs.sweep import (
    ObjectiveConfig,
    PatienceStoppingConfig,
    ThresholdStoppingConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker


def make_outcome(idx: int, value: float | None) -> TrialOutcome:
    return TrialOutcome(trial_id=f"{idx:04d}-x", label=f"trial-{idx}", objective=value)


def test_tracker_records_best_for_maximize() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        early_stopping=None,
    )
    for idx, value in enumerate([0.1, 0.5, 0.3, 0.6]):
        assert tracker.observe(make_outcome(idx, value)) is False

    summary = tracker.summary()
    assert summary.best_value == 0.6
    assert summary.best_trial_id == "0003-x"
    assert summary.halted_by_early_stopping is False


def test_tracker_records_best_for_minimize() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="val/loss", direction="minimize"),
        early_stopping=None,
    )
    for idx, value in enumerate([2.0, 1.5, 1.7, 1.2, 1.4]):
        tracker.observe(make_outcome(idx, value))

    summary = tracker.summary()
    assert summary.best_value == 1.2
    assert summary.best_trial_id == "0003-x"


def test_tracker_threshold_halts_below_floor_when_maximizing() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        ThresholdStoppingConfig(threshold=0.5, min_trials=1),
    )
    assert tracker.observe(make_outcome(0, 0.7)) is False
    assert tracker.observe(make_outcome(1, 0.4)) is True
    assert tracker.summary().halt_reason == "threshold"


def test_tracker_threshold_respects_min_trials() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        ThresholdStoppingConfig(threshold=0.5, min_trials=3),
    )
    assert tracker.observe(make_outcome(0, 0.1)) is False
    assert tracker.observe(make_outcome(1, 0.1)) is False
    # Third trial reaches min_trials and is below threshold → halt.
    assert tracker.observe(make_outcome(2, 0.1)) is True


def test_tracker_patience_halts_after_consecutive_non_improving() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        PatienceStoppingConfig(patience=2, min_trials=1),
    )
    assert tracker.observe(make_outcome(0, 0.5)) is False
    assert tracker.observe(make_outcome(1, 0.4)) is False
    assert tracker.observe(make_outcome(2, 0.3)) is True
    assert tracker.summary().halt_reason == "patience"


def test_tracker_patience_resets_on_improvement() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        PatienceStoppingConfig(patience=2, min_trials=1),
    )
    tracker.observe(make_outcome(0, 0.5))
    tracker.observe(make_outcome(1, 0.4))
    # Improvement resets patience counter.
    tracker.observe(make_outcome(2, 0.7))
    assert tracker.observe(make_outcome(3, 0.6)) is False
    # One more non-improving trial → halt.
    assert tracker.observe(make_outcome(4, 0.5)) is True


def test_tracker_ignores_missing_objective() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        PatienceStoppingConfig(patience=2, min_trials=1),
    )
    tracker.observe(make_outcome(0, 0.5))
    tracker.observe(make_outcome(1, None))
    tracker.observe(make_outcome(2, None))
    summary = tracker.summary()
    assert summary.completed == 1
    assert summary.best_value == 0.5
    assert summary.halted_by_early_stopping is False


def test_tracker_ignores_non_finite_objectives() -> None:
    tracker = TrialOutcomeTracker(
        ObjectiveConfig(metric="reward", direction="maximize"),
        PatienceStoppingConfig(patience=1, min_trials=1),
    )
    tracker.observe(make_outcome(0, 0.5))
    tracker.observe(make_outcome(1, float("nan")))
    tracker.observe(make_outcome(2, float("inf")))

    summary = tracker.summary()
    assert summary.completed == 1
    assert summary.best_value == 0.5
    assert summary.halted_by_early_stopping is False
