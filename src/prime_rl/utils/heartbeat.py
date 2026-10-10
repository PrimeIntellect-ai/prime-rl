import threading
import time

import requests

from prime_rl.configs.shared import HeartbeatConfig
from prime_rl.utils.logger import get_logger

MIN_INTERVAL_S = 30.0


class Heartbeat:
    """Heartbeat monitor that sends heartbeats to Better Stack.

    The beat() method is called on every progress event (train batch, landed eval
    episode, ...), but a ping is only sent when the last one is at least
    ``MIN_INTERVAL_S`` seconds old and none is in flight. Better Stack rate-limits
    frequent pings (HTTP 429), and a monitor only needs pings well within its
    period + grace, so surplus beats are dropped without an HTTP request.

    Args:
        config: The heartbeat config: the Better Stack URL to ping.
    """

    def __init__(self, config: HeartbeatConfig):
        self.heartbeat_url = config.url
        self._lock = threading.Lock()
        self._pending = False
        self._last_beat_at: float | None = None

    def _send_heartbeat(self):
        """Send heartbeat in background thread."""
        try:
            response = requests.get(self.heartbeat_url, timeout=1)
            if response.status_code != 200:
                get_logger().warning(f"BetterStack heartbeat failed with status code: {response.status_code}")
        except requests.RequestException as e:
            get_logger().warning(f"BetterStack heartbeat error: {e}")
        finally:
            with self._lock:
                self._pending = False

    def beat(self):
        """Send a heartbeat.

        Returns immediately without blocking. The HTTP request runs in a daemon
        thread, so even if the server is slow/unresponsive (up to the 1s timeout),
        training continues uninterrupted. The lock is held only briefly
        (microseconds) to check/set flags atomically.

        Beats that arrive while a ping is in flight or within ``MIN_INTERVAL_S`` of
        the last one are dropped; the first beat after the interval sends the
        next ping.
        """
        with self._lock:
            now = time.monotonic()
            if self._pending or (self._last_beat_at is not None and now - self._last_beat_at < MIN_INTERVAL_S):
                return
            self._pending = True
            self._last_beat_at = now
            threading.Thread(target=self._send_heartbeat, daemon=True).start()
