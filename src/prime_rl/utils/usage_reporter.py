"""Reports token usage per training step to the platform API for billing.

Modeled after PrimeMonitor — uses a background thread with an async event loop
for fire-and-forget HTTP POSTs. The platform API is idempotent on
(run_id, step, usage_type), so replays after crash-resume are safe.
"""

import asyncio
import os
from threading import Thread
from typing import Annotated, Any

import httpx
from pydantic import Field

from prime_rl.utils.logger import get_logger

# Import BaseConfig lazily to avoid hard dependency on pydantic_config at import time.
# In production the full config stack is available; for standalone use BaseModel works.
try:
    from prime_rl.utils.config import BaseConfig as _BaseConfig
except ImportError:
    from pydantic import BaseModel as _BaseConfig  # type: ignore[assignment]


class UsageConfig(_BaseConfig):
    """Platform usage reporting configuration."""

    base_url: Annotated[
        str, Field(description="Base URL for the usage API (e.g. https://api.example.com/api/internal/rft).")
    ]
    api_key_var: Annotated[str, Field(description="Environment variable containing the API key.")] = "PRIME_API_KEY"

    @staticmethod
    def from_env() -> "UsageConfig | None":
        """Create config from USAGE_BASE_URL / USAGE_API_KEY_VAR env vars, or None if not set."""
        base_url = os.environ.get("PI_USAGE_BASE_URL")
        if not base_url:
            return None
        return UsageConfig(
            base_url=base_url,
            api_key_var=os.environ.get("PI_USAGE_API_KEY_VAR", "PRIME_API_KEY"),
        )


class UsageReporter:
    """Fire-and-forget token usage reporter.

    Uses a background thread with an async event loop (same pattern as
    PrimeMonitor) so that reporting never blocks the training loop.
    Fork-safe via ``os.register_at_fork``.
    """

    def __init__(self, config: UsageConfig | None):
        self.logger = get_logger()
        self.enabled = False

        if config is None:
            return

        api_key = os.getenv(config.api_key_var)
        if not api_key:
            self.logger.warning("Usage reporter disabled: %s not set", config.api_key_var)
            return

        self._api_key = api_key
        self._base_url = config.base_url
        self.enabled = True

        self._init_async_client()
        os.register_at_fork(after_in_child=self._reinit_after_fork)
        self.logger.info("Usage reporter initialized (base_url=%s)", self._base_url)

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    def report_inference_usage(
        self,
        run_id: str,
        step: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        self._post_usage(
            run_id=run_id,
            step=step,
            usage_type="inference",
            tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def report_training_usage(
        self,
        run_id: str,
        step: int,
        tokens: int,
    ) -> None:
        if not self.enabled:
            return
        self._post_usage(
            run_id=run_id,
            step=step,
            usage_type="training",
            tokens=tokens,
        )

    def close(self) -> None:
        if not self.enabled or not hasattr(self, "_loop"):
            return
        self._flush()

        async def _close():
            await self._client.aclose()

        try:
            future = asyncio.run_coroutine_threadsafe(_close(), self._loop)
            future.result(timeout=5.0)
        except Exception as e:
            self.logger.debug("Error closing usage reporter: %s", e)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def __del__(self) -> None:
        self.close()

    # -- internals --

    def _post_usage(self, **kwargs: Any) -> None:
        future = asyncio.run_coroutine_threadsafe(self._post_async(kwargs), self._loop)
        self._pending_futures.append(future)
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    async def _post_async(self, data: dict[str, Any], max_retries: int = 3) -> None:
        headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}
        url = f"{self._base_url}/usage"

        for attempt in range(max_retries):
            try:
                response = await self._client.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    body = response.json()
                    if body.get("status") == "duplicate":
                        self.logger.debug(
                            "Usage already recorded: run=%s step=%s type=%s",
                            data.get("run_id"),
                            data.get("step"),
                            data.get("usage_type"),
                        )
                    return
                response.raise_for_status()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.warning(
                        "Failed to report usage after %d attempts: %s: %s",
                        max_retries,
                        type(e).__name__,
                        e,
                    )
                else:
                    delay = 2**attempt
                    self.logger.debug(
                        "Retrying usage report in %ds (attempt %d/%d): %s",
                        delay,
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    await asyncio.sleep(delay)

    def _init_async_client(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(timeout=30)
        self._pending_futures: list[asyncio.Future] = []

    def _reinit_after_fork(self) -> None:
        self._init_async_client()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _flush(self, timeout: float = 15.0) -> None:
        if not self._pending_futures:
            return
        self.logger.debug("Flushing %d pending usage report(s)", len(self._pending_futures))
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                self.logger.debug("Pending usage report completed with error: %s", e)
        self._pending_futures.clear()
