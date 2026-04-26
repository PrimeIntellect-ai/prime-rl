from __future__ import annotations

import asyncio
import time
from collections import deque

import wandb
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.logger import get_logger

POLL_INTERVAL = 5.0
WINDOW_SIZE = 20

# Gauge metrics: collected as instantaneous values
GAUGE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
}

# Counter metrics: converted to per-second rates via delta/dt
COUNTER_METRICS = {
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
}

COUNTER_RATE_NAMES = {
    "vllm:prompt_tokens": "prefill_throughput_tps",
    "vllm:generation_tokens": "decode_throughput_tps",
    "vllm:request_success": "completed_requests_per_s",
}

# Histogram metrics: converted to average latency per interval
HISTOGRAM_METRICS = {
    "vllm:nixl_xfer_time_seconds",
}

_COUNTER_TOTAL_TO_NAME = {f"{name}_total": name for name in COUNTER_METRICS}

# Gauges where we log both max and mean across engines (to show imbalance)
_DUAL_AGG_GAUGES = {"vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate"}


def parse_prometheus_text(text: str) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    """Parse Prometheus exposition format into (gauges, counters, histograms).

    For gauge metrics, returns the per-server aggregate (sum for queue sizes,
    max for per-engine metrics like kv cache within a single server).
    """
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    histograms: dict[str, tuple[float, float]] = {}

    for family in text_string_to_metric_families(text):
        if family.type == "gauge" and family.name in GAUGE_METRICS:
            for sample in family.samples:
                if family.name in _DUAL_AGG_GAUGES:
                    gauges[family.name] = max(gauges.get(family.name, 0.0), sample.value)
                else:
                    gauges[family.name] = gauges.get(family.name, 0.0) + sample.value

        elif family.type == "counter" and family.name in COUNTER_METRICS:
            for sample in family.samples:
                counters[family.name] = counters.get(family.name, 0.0) + sample.value

        elif family.name in _COUNTER_TOTAL_TO_NAME:
            canonical = _COUNTER_TOTAL_TO_NAME[family.name]
            for sample in family.samples:
                counters[canonical] = counters.get(canonical, 0.0) + sample.value

        elif family.type == "histogram" and family.name in HISTOGRAM_METRICS:
            h_sum = 0.0
            h_count = 0.0
            for sample in family.samples:
                if sample.name.endswith("_sum"):
                    h_sum += sample.value
                elif sample.name.endswith("_count"):
                    h_count += sample.value
            histograms[family.name] = (h_sum, h_count)

    return gauges, counters, histograms


class InferenceMetricsCollector:
    """Polls vLLM Prometheus /metrics and logs aggregated values to W&B.

    Runs independently of training steps on a time-based axis.
    """

    def __init__(self, admin_clients: list[AsyncClient]):
        self.admin_clients = admin_clients
        self.logger = get_logger()
        self._gauge_history: dict[str, deque[float]] = {}
        self._rate_history: dict[str, deque[float]] = {}
        self._prev_counters: dict[str, tuple[float, float]] = {}
        self._prev_histograms: dict[str, tuple[float, float, float]] = {}
        self._task: asyncio.Task | None = None

    async def start(self):
        wandb.define_metric("inference/*", step_metric="_timestamp")

        async def poll_loop():
            while True:
                try:
                    await self._collect_and_log()
                except Exception as e:
                    self.logger.debug(f"Inference metrics poll failed: {e!r}")
                await asyncio.sleep(POLL_INTERVAL)

        self._task = asyncio.create_task(poll_loop())

    async def _collect_and_log(self):
        now = time.monotonic()

        async def fetch(client: AsyncClient) -> str | None:
            try:
                response = await client.get("/metrics", timeout=5.0)
                response.raise_for_status()
                return response.text
            except Exception as e:
                self.logger.debug(f"Failed to fetch metrics from {client.base_url}: {e!r}")
                return None

        results = await asyncio.gather(*[fetch(client) for client in self.admin_clients])

        # For dual-agg gauges, collect per-server values to compute both max and mean
        dual_agg_values: dict[str, list[float]] = {}
        agg_sum_gauges: dict[str, float] = {}
        agg_counters: dict[str, float] = {}
        agg_histograms: dict[str, tuple[float, float]] = {}
        n_servers = 0

        for text in results:
            if text is None:
                continue
            n_servers += 1
            gauges, counters, histograms = parse_prometheus_text(text)

            for name, value in gauges.items():
                if name in _DUAL_AGG_GAUGES:
                    dual_agg_values.setdefault(name, []).append(value)
                else:
                    agg_sum_gauges[name] = agg_sum_gauges.get(name, 0.0) + value

            for name, value in counters.items():
                agg_counters[name] = agg_counters.get(name, 0.0) + value

            for name, (h_sum, h_count) in histograms.items():
                prev = agg_histograms.get(name, (0.0, 0.0))
                agg_histograms[name] = (prev[0] + h_sum, prev[1] + h_count)

        if n_servers == 0:
            return

        # Update gauge history — sum gauges
        for name, value in agg_sum_gauges.items():
            short = name.removeprefix("vllm:")
            if short not in self._gauge_history:
                self._gauge_history[short] = deque(maxlen=WINDOW_SIZE)
            self._gauge_history[short].append(value)

        # Update gauge history — dual-agg gauges (max + mean across engines)
        for name, values in dual_agg_values.items():
            short = name.removeprefix("vllm:")
            max_key = f"{short}_max"
            mean_key = f"{short}_mean"
            if max_key not in self._gauge_history:
                self._gauge_history[max_key] = deque(maxlen=WINDOW_SIZE)
            if mean_key not in self._gauge_history:
                self._gauge_history[mean_key] = deque(maxlen=WINDOW_SIZE)
            self._gauge_history[max_key].append(max(values))
            self._gauge_history[mean_key].append(sum(values) / len(values))

        # Compute rates from counters
        for name, value in agg_counters.items():
            rate_name = COUNTER_RATE_NAMES[name]
            prev = self._prev_counters.get(name)
            self._prev_counters[name] = (now, value)
            if prev is None:
                continue
            prev_time, prev_value = prev
            dt = now - prev_time
            if dt <= 0:
                continue
            delta = value - prev_value
            if delta < 0:
                continue
            rate = delta / dt
            if rate_name not in self._rate_history:
                self._rate_history[rate_name] = deque(maxlen=WINDOW_SIZE)
            self._rate_history[rate_name].append(rate)

        # Compute average histogram latency
        for name, (h_sum, h_count) in agg_histograms.items():
            short = name.removeprefix("vllm:")
            rate_name = f"{short}_avg_ms"
            prev = self._prev_histograms.get(name)
            self._prev_histograms[name] = (now, h_sum, h_count)
            if prev is None:
                continue
            prev_time, prev_sum, prev_count = prev
            d_sum = h_sum - prev_sum
            d_count = h_count - prev_count
            if d_count < 0 or d_sum < 0:
                continue
            if d_count > 0:
                avg_ms = (d_sum / d_count) * 1000.0
                if rate_name not in self._rate_history:
                    self._rate_history[rate_name] = deque(maxlen=WINDOW_SIZE)
                self._rate_history[rate_name].append(avg_ms)

        # Build smoothed metrics dict
        metrics: dict[str, float] = {}
        for short, values in self._gauge_history.items():
            if values:
                metrics[f"inference/{short}"] = sum(values) / len(values)
        for rate_name, values in self._rate_history.items():
            if values:
                metrics[f"inference/{rate_name}"] = sum(values) / len(values)

        if metrics:
            metrics["_timestamp"] = time.time()
            wandb.log(metrics)

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
