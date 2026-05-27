from __future__ import annotations

import asyncio
import re
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
        self._server_gauge_history: dict[str, dict[str, deque[float]]] = {}
        self._server_rate_history: dict[str, dict[str, deque[float]]] = {}
        self._server_prev_counters: dict[str, dict[str, tuple[float, float]]] = {}
        self._server_prev_histograms: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._server_names = [self._server_name(idx, client) for idx, client in enumerate(admin_clients)]
        self._task: asyncio.Task | None = None

    @staticmethod
    def _server_name(idx: int, client: AsyncClient) -> str:
        host = client.base_url.host or f"server_{idx}"
        port = client.base_url.port
        raw = f"{host}_{port}" if port is not None else host
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
        return f"server_{idx:02d}_{safe or 'unknown'}"

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
        active_servers: set[str] = set()
        n_servers = 0

        for server_name, text in zip(self._server_names, results, strict=True):
            if text is None:
                continue
            active_servers.add(server_name)
            n_servers += 1
            gauges, counters, histograms = parse_prometheus_text(text)
            self._update_server_histories(server_name, now, gauges, counters, histograms)

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
            wandb.log({**self._server_up_metrics(active_servers), "_timestamp": time.time()})
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
        self._add_cache_alias_metrics(metrics)
        metrics.update(self._server_metrics(active_servers))
        metrics.update(self._server_up_metrics(active_servers))

        if metrics:
            metrics["_timestamp"] = time.time()
            wandb.log(metrics)

    def _update_server_histories(
        self,
        server_name: str,
        now: float,
        gauges: dict[str, float],
        counters: dict[str, float],
        histograms: dict[str, tuple[float, float]],
    ) -> None:
        gauge_history = self._server_gauge_history.setdefault(server_name, {})
        rate_history = self._server_rate_history.setdefault(server_name, {})
        prev_counters = self._server_prev_counters.setdefault(server_name, {})
        prev_histograms = self._server_prev_histograms.setdefault(server_name, {})

        for name, value in gauges.items():
            short = name.removeprefix("vllm:")
            if name in _DUAL_AGG_GAUGES:
                short = f"{short}_max"
            gauge_history.setdefault(short, deque(maxlen=WINDOW_SIZE)).append(value)

        for name, value in counters.items():
            rate_name = COUNTER_RATE_NAMES[name]
            prev = prev_counters.get(name)
            prev_counters[name] = (now, value)
            if prev is None:
                continue
            prev_time, prev_value = prev
            dt = now - prev_time
            if dt <= 0:
                continue
            delta = value - prev_value
            if delta < 0:
                continue
            rate_history.setdefault(rate_name, deque(maxlen=WINDOW_SIZE)).append(delta / dt)

        for name, (h_sum, h_count) in histograms.items():
            short = name.removeprefix("vllm:")
            rate_name = f"{short}_avg_ms"
            prev = prev_histograms.get(name)
            prev_histograms[name] = (now, h_sum, h_count)
            if prev is None:
                continue
            _, prev_sum, prev_count = prev
            d_sum = h_sum - prev_sum
            d_count = h_count - prev_count
            if d_count < 0 or d_sum < 0:
                continue
            if d_count > 0:
                rate_history.setdefault(rate_name, deque(maxlen=WINDOW_SIZE)).append((d_sum / d_count) * 1000.0)

    def _server_metrics(self, active_servers: set[str]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for server_name, gauge_history in self._server_gauge_history.items():
            if server_name not in active_servers:
                continue
            for short, values in gauge_history.items():
                if values:
                    metrics[f"inference/server/{server_name}/{short}"] = sum(values) / len(values)
        for server_name, rate_history in self._server_rate_history.items():
            if server_name not in active_servers:
                continue
            for rate_name, values in rate_history.items():
                if values:
                    metrics[f"inference/server/{server_name}/{rate_name}"] = sum(values) / len(values)
        self._add_cache_alias_metrics(metrics)
        return metrics

    def _server_up_metrics(self, active_servers: set[str]) -> dict[str, float]:
        return {f"inference/server/{server_name}/up": float(server_name in active_servers) for server_name in self._server_names}

    @classmethod
    def _add_cache_alias_metrics(cls, metrics: dict[str, float]) -> None:
        for key, value in list(metrics.items()):
            if key.endswith("/gpu_prefix_cache_hit_rate_max"):
                prefix = key.removesuffix("gpu_prefix_cache_hit_rate_max")
                metrics[f"{prefix}kv_cache_hit_rate_max"] = value
            elif key.endswith("/gpu_prefix_cache_hit_rate_mean"):
                prefix = key.removesuffix("gpu_prefix_cache_hit_rate_mean")
                metrics[f"{prefix}kv_cache_hit_rate_mean"] = value
            elif key.endswith("/gpu_cache_usage_perc_max"):
                prefix = key.removesuffix("gpu_cache_usage_perc_max")
                metrics[f"{prefix}kv_cache_left_perc_min"] = cls._cache_left(value)
            elif key.endswith("/gpu_cache_usage_perc_mean"):
                prefix = key.removesuffix("gpu_cache_usage_perc_mean")
                metrics[f"{prefix}kv_cache_left_perc_mean"] = cls._cache_left(value)

    @staticmethod
    def _cache_left(usage: float) -> float:
        return min(max(1.0 - usage, 0.0), 1.0)

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
