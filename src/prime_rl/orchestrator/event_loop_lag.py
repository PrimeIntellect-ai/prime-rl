import asyncio
from time import perf_counter

import numpy as np

from prime_rl.utils.logger import get_logger


class EventLoopLagMonitor:
    """A class to monitor how busy the main event loop is."""

    def __init__(
        self,
        interval: float = 1.0,
        max_window_size: int = 10000,
        warn_med_lag_threshold: float = 0.5,
        warn_p90_lag_threshold: float = 1.0,
        warn_max_lag_threshold: float = 5.0,
    ):
        assert (
            interval > 0
            and max_window_size > 0
            and warn_max_lag_threshold > 0
            and warn_med_lag_threshold > 0
            and warn_p90_lag_threshold > 0
        )
        self.interval = interval
        self.max_window_size = max_window_size
        self.warn_max_lag_threshold = warn_max_lag_threshold
        self.warn_med_lag_threshold = warn_med_lag_threshold
        self.warn_p90_lag_threshold = warn_p90_lag_threshold
        self.logger = get_logger()
        self.lags = []

    async def measure_lag(self):
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        next_time = perf_counter() + self.interval
        await asyncio.sleep(self.interval)
        now = perf_counter()
        lag = now - next_time
        return lag

    async def run(self):
        """Infinite loop to periodically measure event loop lag. Should be started as background task."""
        while True:
            lag = await self.measure_lag()
            self.lags.append(lag)
            if len(self.lags) > self.max_window_size:
                self.lags.pop(0)

    def reset(self):
        """Reset the list of measured lags."""
        self.lags = []

    def get_metrics(self) -> dict[str, float]:
        """Compute metrics for the event loop lag over the last window_size measurements."""
        window_size = int(min(self.max_window_size, len(self.lags)))
        if window_size <= 0:
            return {}
        last_lags = np.array(self.lags[-window_size:])
        mean_lag = float(np.mean(last_lags))
        med_lag = float(np.median(last_lags))
        p90_lag = float(np.percentile(last_lags, 90))
        min_lag = float(np.min(last_lags))
        max_lag = float(np.max(last_lags))
        if (
            med_lag > self.warn_med_lag_threshold
            or p90_lag > self.warn_p90_lag_threshold
            or max_lag > self.warn_max_lag_threshold
        ):
            self.logger.warning(
                f"Detected busy event loop. Measured {mean_lag:.1f}s (min={min_lag:.1f}s, med={med_lag:.1f}s, p90={p90_lag:.1f}s, max={max_lag:.1f}s) event loop lag over the last {len(last_lags)} measurement(s)"
            )

        return {
            "event_loop_lag/min": min_lag,
            "event_loop_lag/mean": mean_lag,
            "event_loop_lag/med": med_lag,
            "event_loop_lag/p90": p90_lag,
            "event_loop_lag/max": max_lag,
        }
