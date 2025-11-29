import threading
import time
from typing import Optional

import requests

from prime_rl.utils.logger import get_logger


class Heartbeat:
    """Heartbeat monitor that sends heartbeats at specified intervals.

    The step() method can be called every training step, but will only send
    a heartbeat to Better Stack if the specified time interval has elapsed
    since the last heartbeat was sent.

    Args:
        heartbeat_url: The unique URL provided by Better Stack for the heartbeat.
        interval_seconds: Time interval in seconds between heartbeats.
    """

    def __init__(self, heartbeat_url: str, interval_seconds: float):
        self.heartbeat_url = heartbeat_url
        self.interval_seconds = interval_seconds
        self.last_sent_time: Optional[float] = None
        self._lock = threading.Lock()
        self._pending = False

    def _send_heartbeat(self, send_time: float):
        """Send heartbeat in background thread."""
        try:
            response = requests.get(self.heartbeat_url, timeout=1)
            if response.status_code == 200:
                with self._lock:
                    self.last_sent_time = send_time
                    self._pending = False
            else:
                get_logger().warning(f"BetterStack heartbeat failed with status code: {response.status_code}")
                with self._lock:
                    self._pending = False
        except requests.RequestException as e:
            get_logger().warning(f"BetterStack heartbeat error: {e}")
            with self._lock:
                self._pending = False

    def step(self) -> bool:
        """Send a heartbeat if the specified time interval has passed.

        Returns immediately without blocking. Heartbeat is sent in background thread.

        Non-blocking guarantee: This method never blocks the training loop. The HTTP
        request runs in a daemon thread, so even if the server is slow/unresponsive
        (up to the 2s timeout), training continues uninterrupted. The lock is held only
        briefly (microseconds) to check/set flags atomically.

        Returns:
            True if a heartbeat was triggered, False otherwise.
        """
        current_time = time.perf_counter()

        with self._lock:
            # Send heartbeat if interval has elapsed or if this is the first call
            if (
                self.last_sent_time is None or current_time - self.last_sent_time >= self.interval_seconds
            ) and not self._pending:
                self._pending = True
                thread = threading.Thread(target=self._send_heartbeat, args=(current_time,), daemon=True)
                thread.start()
                return True

        return False
