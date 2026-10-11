from prime_rl.trainer.utils import SpikeGuard


def test_spike_guard_flags_values_above_factor_times_median_once_the_window_is_full():
    guard = SpikeGuard(factor=10.0, window=3)
    assert not any(guard.is_spike(value) for value in [1.0, 2.0, 100.0])  # window not full yet
    assert guard.is_spike(21.0)  # 21 > 10 x median(1, 2, 100)
    assert not guard.is_spike(20.0)
    assert list(guard.history) == [2.0, 100.0, 20.0]  # flagged values stay out of the median
