import pytest

from prime_rl.orchestrator.trainer_capacity import TrainerCapacity


def test_capacity_is_rollouts_over_busy_time():
    capacity = TrainerCapacity()
    # Every 100 s a batch of 64 ships; the trainer works 40 s of each step and waits 60 s for its next batch
    for step in range(6):
        shipped = 100.0 * step
        capacity.shipped(64, requested_at=shipped - 60 if step else None, now=shipped)
    assert capacity.rate == pytest.approx(64 / 40)  # the two warmup ships are left out
    assert capacity.busy == pytest.approx(0.4)
    assert capacity.waited == pytest.approx(0.6)
    capacity.shipped(64, requested_at=None, now=600.0)  # an unreported step is skipped
    assert len(capacity.steps) == 4
