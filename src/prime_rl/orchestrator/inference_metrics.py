"""Snapshot vLLM Prometheus /metrics on demand.

Counter metrics are converted to per-second rates via delta/dt against the
previous snapshot. Polled lazily from the batcher per shipped step so we
don't need a separate background task.
"""

import time
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.logger import get_logger

GAUGE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
}

COUNTER_METRICS = {
    "vllm:prompt_tokens": "prefill_throughput_tps",
    "vllm:generation_tokens": "decode_throughput_tps",
    "vllm:request_success": "completed_requests_per_s",
}

# Histograms tracked as average since last snapshot (sum delta / count delta).
HISTOGRAM_METRICS = {
    "vllm:nixl_xfer_time_seconds": "nixl_xfer_avg_ms",
}

# Counter families show up as either `<name>` or `<name>_total` in the
# exposition; map both to the canonical name.
_COUNTER_TOTAL_TO_NAME = {f"{name}_total": name for name in COUNTER_METRICS}


def _parse(text: str) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    histograms: dict[str, tuple[float, float]] = {}
    for family in text_string_to_metric_families(text):
        if family.type == "gauge" and family.name in GAUGE_METRICS:
            for s in family.samples:
                # Sum across engines/workers (e.g. queue sizes), max for cache
                # gauges to surface the most-loaded engine.
                if family.name in {"vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate"}:
                    gauges[family.name] = max(gauges.get(family.name, 0.0), s.value)
                else:
                    gauges[family.name] = gauges.get(family.name, 0.0) + s.value
        elif family.type == "counter" and family.name in COUNTER_METRICS:
            for s in family.samples:
                counters[family.name] = counters.get(family.name, 0.0) + s.value
        elif family.name in _COUNTER_TOTAL_TO_NAME:
            canonical = _COUNTER_TOTAL_TO_NAME[family.name]
            for s in family.samples:
                counters[canonical] = counters.get(canonical, 0.0) + s.value
        elif family.type == "histogram" and family.name in HISTOGRAM_METRICS:
            h_sum = 0.0
            h_count = 0.0
            for s in family.samples:
                if s.name.endswith("_sum"):
                    h_sum += s.value
                elif s.name.endswith("_count"):
                    h_count += s.value
            histograms[family.name] = (h_sum, h_count)
    return gauges, counters, histograms


class InferenceMetricsCollector:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.logger = get_logger()
        self._prev_counters: dict[str, tuple[float, float]] = {}
        self._prev_histograms: dict[str, tuple[float, float, float]] = {}

    async def collect(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/metrics", timeout=5.0)
            r.raise_for_status()
            text = r.text
        except (httpx.HTTPError, TimeoutError) as e:
            self.logger.debug(f"inference /metrics fetch failed: {e!r}")
            return {}
        gauges, counters, histograms = _parse(text)
        now = time.monotonic()
        out: dict[str, Any] = {}

        for name, value in gauges.items():
            out[f"inference/{name.removeprefix('vllm:')}"] = value

        for name, value in counters.items():
            prev = self._prev_counters.get(name)
            self._prev_counters[name] = (now, value)
            if prev is None:
                continue
            dt = now - prev[0]
            if dt <= 0:
                continue
            delta = value - prev[1]
            if delta < 0:
                continue  # vLLM restart
            out[f"inference/{COUNTER_METRICS[name]}"] = delta / dt

        for name, (h_sum, h_count) in histograms.items():
            prev = self._prev_histograms.get(name)
            self._prev_histograms[name] = (now, h_sum, h_count)
            if prev is None:
                continue
            d_sum = h_sum - prev[1]
            d_count = h_count - prev[2]
            if d_count <= 0 or d_sum < 0:
                continue
            out[f"inference/{HISTOGRAM_METRICS[name]}"] = (d_sum / d_count) * 1000.0

        return out
