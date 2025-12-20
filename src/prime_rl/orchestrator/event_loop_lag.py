import asyncio
from time import perf_counter

from prime_rl.utils.logger import get_logger


class EventLoopLagMonitor:
    """A class to monitor how busy the main event loop is."""

    def __init__(self, interval: float = 1.0, warn_lag_threshold: float = 1.0, max_window_size: int = 1000):
        assert interval > 0 and warn_lag_threshold > 0 and max_window_size > 0
        self.interval = interval
        self.warn_lag_threshold = warn_lag_threshold
        self.max_window_size = max_window_size
        self.logger = get_logger()
        self.lags = []

    async def measure_lag(self, interval: float | None = None):
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        if interval is None:
            interval = self.interval
        assert interval > 0
        next_time = perf_counter() + interval
        await asyncio.sleep(interval)
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

    def get_avg_lag(self, window_size: int | None = None) -> float:
        """Get the average event loop lag over the last window_size measurements."""
        if window_size is None:
            window_size = self.max_window_size
        assert window_size > 0
        avg_lag = sum(self.lags[-window_size:]) / min(window_size, len(self.lags))
        if avg_lag > self.warn_lag_threshold:
            self.logger.warning(
                f"Detected busy event loop. Measured {avg_lag:.1f}s event loop lag over the last {window_size} measurement(s)"
            )
        return avg_lag

    def reset(self):
        self.lags = []
