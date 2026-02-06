"""Billing client for reporting token usage to Prime Intellect API."""

import asyncio
import os
from threading import Thread
from typing import Any

import httpx

from prime_rl.utils.logger import get_logger


class BillingClient:
    """
    Lightweight client for reporting token usage to the billing API.

    This is used by the trainer/packer to report per-run token usage.
    Unlike PrimeMonitor, this doesn't require RUN_ID environment variable
    since it supports multiple runs with different run_ids.
    """

    def __init__(
        self,
        base_url: str = "https://api.primeintellect.ai/api/internal/rft",
        api_key_var: str = "PRIME_API_KEY",
    ):
        self.logger = get_logger()
        self.base_url = base_url

        # Only enable on rank 0
        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.is_master = rank == 0
        if not self.is_master:
            self.enabled = False
            return

        # Get API key from environment variable
        api_key = os.getenv(api_key_var)
        if not api_key:
            self.logger.debug(f"Billing API key not found ({api_key_var}). Usage reporting disabled.")
            self.enabled = False
            return

        self.api_key = api_key
        self.enabled = True

        # Set up async HTTP client with background event loop
        self._init_async_client()
        os.register_at_fork(after_in_child=self._reinit_after_fork)

    def _init_async_client(self) -> None:
        """Initialize the event loop, background thread, and HTTP client."""
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(timeout=30)
        self._pending_futures: list[asyncio.Future] = []

    def _reinit_after_fork(self) -> None:
        """Reinitialize thread and event loop after fork."""
        self._init_async_client()

    def _run_event_loop(self) -> None:
        """Run the async event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _make_request_async(self, data: dict[str, Any], max_retries: int = 3) -> None:
        """Make an async POST request to the usage endpoint with retries."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/usage"

        for attempt in range(max_retries):
            try:
                response = await self._client.post(endpoint, headers=headers, json=data)
                response.raise_for_status()
                return
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    self.logger.debug(f"Failed to report usage after {max_retries} attempts: {type(e).__name__}: {e}")
                else:
                    delay = 2**attempt
                    await asyncio.sleep(delay)

    def report_usage(
        self,
        run_id: str,
        step: int,
        tokens: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        usage_type: str = "training",
    ) -> None:
        """
        Report token usage for billing.

        Args:
            run_id: The run ID (from folder name like run_xxx)
            step: Current training step
            tokens: Total tokens processed
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            usage_type: Either "training" or "inference"
        """
        if not self.enabled:
            return

        payload = {
            "run_id": run_id,
            "step": step,
            "usage_type": usage_type,
            "tokens": tokens,
        }

        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens

        future = asyncio.run_coroutine_threadsafe(self._make_request_async(payload), self._loop)
        self._pending_futures.append(future)
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def close(self) -> None:
        """Close the HTTP client and stop the background event loop."""
        if not hasattr(self, "_client"):
            return

        async def _close_client() -> None:
            await self._client.aclose()

        try:
            future = asyncio.run_coroutine_threadsafe(_close_client(), self._loop)
            future.result(timeout=5.0)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


# Global billing client instance
_BILLING_CLIENT: BillingClient | None = None


def get_billing_client() -> BillingClient:
    """Get or create the global billing client."""
    global _BILLING_CLIENT
    if _BILLING_CLIENT is None:
        _BILLING_CLIENT = BillingClient()
    return _BILLING_CLIENT
