from __future__ import annotations

import asyncio
from collections import deque

from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger


def parse_prometheus_text(text: str, metric_names: set[str]) -> dict[tuple[str, str], float]:
    """Parse Prometheus exposition format text and extract gauge values for the given metric names.

    Returns a dict keyed by (metric_name, engine_index) to preserve per-engine granularity.
    If no engine label is present, engine defaults to "0".
    """
    results: dict[tuple[str, str], float] = {}
    for family in text_string_to_metric_families(text):
        if family.type != "gauge" or family.name not in metric_names:
            continue
        for sample in family.samples:
            engine = sample.labels.get("engine", "0")
            results[(family.name, engine)] = sample.value
    return results


# Key: (server_url, metric_name, engine_index)
HistoryKey = tuple[str, str, str]


class InferenceMetricsCollector:
    """Polls Prometheus /metrics from inference servers and exposes per-server metrics.

    Reads the current admin_clients from the inference pool on each poll, so it
    automatically adapts when servers join or leave (elastic pools).

    Metrics are collected in the background every `poll_interval` seconds.
    `get_metrics()` returns the running average over the last `window_size` fetches.
    """

    METRICS = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    }

    def __init__(self, inference_pool: InferencePool, poll_interval: float = 5.0, window_size: int = 20):
        self.inference_pool = inference_pool
        self.poll_interval = poll_interval
        self.window_size = window_size
        self.logger = get_logger()
        self._history: dict[HistoryKey, deque[float]] = {}
        self._active_urls: set[str] = set()
        self._task: asyncio.Task | None = None

    async def start(self):
        async def poll_loop():
            while True:
                await self.collect()
                await asyncio.sleep(self.poll_interval)

        self._task = asyncio.create_task(poll_loop())

    async def collect(self):
        """Fetch /metrics from all servers concurrently and append to history."""
        admin_clients = self.inference_pool.admin_clients

        async def fetch(client: AsyncClient) -> tuple[str, dict[tuple[str, str], float] | None]:
            url = str(client.base_url)
            try:
                response = await client.get("/metrics", timeout=5.0)
                response.raise_for_status()
                return url, parse_prometheus_text(response.text, self.METRICS)
            except Exception as e:
                self.logger.debug(f"Failed to fetch metrics from {url}: {e!r}")
                return url, None

        results = await asyncio.gather(*[fetch(client) for client in admin_clients])

        # Clear stale entries from servers that left the pool
        active_urls = {str(client.base_url) for client in admin_clients}
        for key in list(self._history):
            if key[0] not in active_urls:
                del self._history[key]
        self._active_urls = active_urls

        for url, metrics in results:
            if metrics is None:
                continue
            for (metric_name, engine), value in metrics.items():
                key = (url, metric_name, engine)
                if key not in self._history:
                    self._history[key] = deque(maxlen=self.window_size)
                self._history[key].append(value)

    def get_metrics(self) -> dict[str, float]:
        """Return per-server, per-engine running averages formatted for W&B logging.

        Keys are structured as `inference/<metric_name>/server_<idx>_engine_<idx>`
        """
        metrics: dict[str, float] = {}
        sorted_urls = sorted(self._active_urls)
        url_to_idx = {url: i for i, url in enumerate(sorted_urls)}

        for (url, metric_name, engine), values in self._history.items():
            if url not in url_to_idx or not values:
                continue
            server_idx = url_to_idx[url]
            short_name = metric_name.removeprefix("vllm:")
            avg = sum(values) / len(values)
            metrics[f"inference/{short_name}/server_{server_idx}_engine_{engine}"] = avg
        return metrics

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
