import threading

import requests

from prime_rl.utils.logger import get_logger


class Heartbeat:
    """Heartbeat monitor that sends heartbeats to Better Stack.

    The beat() method sends a heartbeat every time it's called. Heartbeats are
    sent in a background thread to avoid blocking the training loop.

    Args:
        heartbeat_url: The unique URL provided by Better Stack for the heartbeat.
    """

    def __init__(self, heartbeat_url: str):
        self.heartbeat_url = heartbeat_url
        self._lock = threading.Lock()
        self._pending = False

    def _send_heartbeat(self):
        """Send heartbeat in background thread."""
        try:
            response = requests.get(self.heartbeat_url, timeout=1)
            if response.status_code == 200:
                with self._lock:
                    self._pending = False
            else:
                get_logger().warning(f"BetterStack heartbeat failed with status code: {response.status_code}")
                with self._lock:
                    self._pending = False
        except requests.RequestException as e:
            get_logger().warning(f"BetterStack heartbeat error: {e}")
            with self._lock:
                self._pending = False

    def beat(self):
        """Send a heartbeat.

        Returns immediately without blocking. Heartbeat is sent in background thread.

        Non-blocking guarantee: This method never blocks the training loop. The HTTP
        request runs in a daemon thread, so even if the server is slow/unresponsive
        (up to the 2s timeout), training continues uninterrupted. The lock is held only
        briefly (microseconds) to check/set flags atomically.
        """
        with self._lock:
            if not self._pending:
                self._pending = True
                thread = threading.Thread(target=self._send_heartbeat, daemon=True)
                thread.start()
