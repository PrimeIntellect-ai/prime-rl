from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import wandb
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.logger import get_logger

POLL_INTERVAL = 5.0
WINDOW_SIZE = 20
PD_ROLES = {"prefill", "decode"}

COUNTER_KEYS = {
    "vllm:prompt_tokens": "prompt_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens": "generation_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success": "request_success_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:prefix_cache_queries": "prefix_cache_queries",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:prefix_cache_hits": "prefix_cache_hits",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:nixl_num_failed_transfers_total": "nixl_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total": "nixl_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total": "nixl_kv_expired_requests_total",
}

GAUGE_KEYS = {
    "vllm:num_requests_running": "running_requests",
    "vllm:num_requests_waiting": "waiting_requests",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate": "gpu_prefix_cache_hit_rate",
    "vllm:cpu_prefix_cache_hit_rate": "cpu_prefix_cache_hit_rate",
}

HISTOGRAM_SUM_KEYS = {
    "vllm:request_prefill_time_seconds_sum": "request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum": "request_decode_time_seconds_sum",
    "vllm:request_queue_time_seconds_sum": "request_queue_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum": "time_to_first_token_seconds_sum",
    "vllm:inter_token_latency_seconds_sum": "inter_token_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum": "e2e_request_latency_seconds_sum",
    "vllm:nixl_xfer_time_seconds_sum": "nixl_xfer_time_seconds_sum",
    "vllm:nixl_bytes_transferred_sum": "nixl_bytes_transferred_sum",
}

HISTOGRAM_COUNT_KEYS = {
    "vllm:request_prefill_time_seconds_count": "request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_count": "request_decode_time_seconds_count",
    "vllm:request_queue_time_seconds_count": "request_queue_time_seconds_count",
    "vllm:time_to_first_token_seconds_count": "time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_count": "inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_count": "e2e_request_latency_seconds_count",
    "vllm:nixl_xfer_time_seconds_count": "nixl_xfer_time_seconds_count",
    "vllm:nixl_bytes_transferred_count": "nixl_bytes_transferred_count",
}


@dataclass
class EngineRollup:
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    kv_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float | None = None
    gpu_prefix_cache_hit_rate: float | None = None
    cpu_prefix_cache_hit_rate: float | None = None
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    request_success_total: float = 0.0
    prefix_cache_queries: float = 0.0
    prefix_cache_hits: float = 0.0
    nixl_failed_transfers_total: float = 0.0
    nixl_failed_notifications_total: float = 0.0
    nixl_kv_expired_requests_total: float = 0.0
    request_prefill_time_seconds_sum: float = 0.0
    request_prefill_time_seconds_count: float = 0.0
    request_decode_time_seconds_sum: float = 0.0
    request_decode_time_seconds_count: float = 0.0
    request_queue_time_seconds_sum: float = 0.0
    request_queue_time_seconds_count: float = 0.0
    time_to_first_token_seconds_sum: float = 0.0
    time_to_first_token_seconds_count: float = 0.0
    inter_token_latency_seconds_sum: float = 0.0
    inter_token_latency_seconds_count: float = 0.0
    e2e_request_latency_seconds_sum: float = 0.0
    e2e_request_latency_seconds_count: float = 0.0
    nixl_xfer_time_seconds_sum: float = 0.0
    nixl_xfer_time_seconds_count: float = 0.0
    nixl_bytes_transferred_sum: float = 0.0
    nixl_bytes_transferred_count: float = 0.0


@dataclass
class NodeRollup:
    engines: dict[str, EngineRollup] = field(default_factory=dict)

    @property
    def engine_count(self) -> int:
        return len(self.engines)

    def summed(self, attribute: str) -> float:
        return sum(getattr(engine, attribute) for engine in self.engines.values())

    def values(self, attribute: str) -> list[float]:
        return [value for engine in self.engines.values() if (value := getattr(engine, attribute)) is not None]


@dataclass(frozen=True)
class MetricsEndpoint:
    client: AsyncClient
    role: str | None
    key: str


@dataclass(frozen=True)
class TimedRollup:
    timestamp: float
    rollup: NodeRollup


@dataclass(frozen=True)
class EndpointSample:
    endpoint: MetricsEndpoint
    timestamp: float
    rollup: NodeRollup


def parse_prometheus_text(text: str) -> NodeRollup:
    """Parse vLLM Prometheus metrics into per-engine counters, gauges, and histogram totals."""
    engines: dict[str, EngineRollup] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            engine_id = sample.labels.get("engine", "aggregate")
            engine = engines.setdefault(engine_id, EngineRollup())
            if sample.name in GAUGE_KEYS:
                setattr(engine, GAUGE_KEYS[sample.name], float(sample.value))
            elif sample.name in COUNTER_KEYS:
                attribute = COUNTER_KEYS[sample.name]
                setattr(engine, attribute, getattr(engine, attribute) + float(sample.value))
            elif sample.name in HISTOGRAM_SUM_KEYS:
                setattr(engine, HISTOGRAM_SUM_KEYS[sample.name], float(sample.value))
            elif sample.name in HISTOGRAM_COUNT_KEYS:
                setattr(engine, HISTOGRAM_COUNT_KEYS[sample.name], float(sample.value))
    return NodeRollup(engines=engines)


def build_metrics_endpoints(
    admin_clients: list[AsyncClient], roles: list[str | None] | None = None
) -> list[MetricsEndpoint]:
    """Attach optional P/D roles to admin clients used for metrics polling."""
    if roles is None:
        roles = [None] * len(admin_clients)
    if len(roles) != len(admin_clients):
        raise ValueError(f"Got {len(roles)} inference metric role(s) for {len(admin_clients)} admin client(s)")

    endpoints: list[MetricsEndpoint] = []
    for client, role in zip(admin_clients, roles):
        normalized_role = role if role in PD_ROLES else None
        if role is not None and normalized_role is None:
            raise ValueError(f"Unsupported inference metrics role: {role}")
        endpoints.append(MetricsEndpoint(client=client, role=normalized_role, key=str(client.base_url).rstrip("/")))
    return endpoints


def max_vio(values: list[float]) -> float:
    """Return the max-load violation used for MoE load-balance reporting."""
    if not values:
        return 0.0
    balanced_load = sum(values) / len(values)
    if balanced_load == 0:
        return 0.0
    return (max(values) - balanced_load) / balanced_load


def counter_rate(samples: list[EndpointSample], previous: dict[str, TimedRollup], attribute: str) -> float | None:
    """Sum per-endpoint counter rates for a scope."""
    rates: list[float] = []
    for sample in samples:
        previous_sample = previous.get(sample.endpoint.key)
        if previous_sample is None:
            continue
        dt = sample.timestamp - previous_sample.timestamp
        delta = sample.rollup.summed(attribute) - previous_sample.rollup.summed(attribute)
        if dt <= 0 or delta < 0:
            continue
        rates.append(delta / dt)
    if not rates:
        return None
    return sum(rates)


def counter_ratio(
    samples: list[EndpointSample],
    previous: dict[str, TimedRollup],
    numerator_attribute: str,
    denominator_attribute: str,
) -> float | None:
    """Compute a scope-level ratio from counter deltas."""
    numerator_delta = 0.0
    denominator_delta = 0.0
    for sample in samples:
        previous_sample = previous.get(sample.endpoint.key)
        if previous_sample is None:
            continue
        endpoint_numerator_delta = sample.rollup.summed(numerator_attribute) - previous_sample.rollup.summed(
            numerator_attribute
        )
        endpoint_denominator_delta = sample.rollup.summed(denominator_attribute) - previous_sample.rollup.summed(
            denominator_attribute
        )
        if endpoint_numerator_delta < 0 or endpoint_denominator_delta < 0:
            continue
        numerator_delta += endpoint_numerator_delta
        denominator_delta += endpoint_denominator_delta
    if denominator_delta <= 0:
        return None
    return numerator_delta / denominator_delta


def histogram_average(
    samples: list[EndpointSample],
    previous: dict[str, TimedRollup],
    sum_attribute: str,
    count_attribute: str,
) -> float | None:
    """Compute a scope-level histogram average from sum/count deltas."""
    sum_delta = 0.0
    count_delta = 0.0
    for sample in samples:
        previous_sample = previous.get(sample.endpoint.key)
        if previous_sample is None:
            continue
        endpoint_sum_delta = sample.rollup.summed(sum_attribute) - previous_sample.rollup.summed(sum_attribute)
        endpoint_count_delta = sample.rollup.summed(count_attribute) - previous_sample.rollup.summed(count_attribute)
        if endpoint_sum_delta < 0 or endpoint_count_delta < 0:
            continue
        sum_delta += endpoint_sum_delta
        count_delta += endpoint_count_delta
    if count_delta <= 0:
        return None
    return sum_delta / count_delta


def mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for non-empty metric value lists."""
    if not values:
        return None
    return sum(values) / len(values)


def build_scope_metrics(
    scope: str,
    samples: list[EndpointSample],
    previous: dict[str, TimedRollup],
) -> dict[str, float]:
    """Build W&B metric values for one inference scope."""
    running_values = [value for sample in samples for value in sample.rollup.values("running_requests")]
    waiting_values = [value for sample in samples for value in sample.rollup.values("waiting_requests")]
    kv_values = [value for sample in samples for value in sample.rollup.values("kv_cache_usage_perc")]
    cpu_kv_values = [value for sample in samples for value in sample.rollup.values("cpu_cache_usage_perc")]
    prefix_gauge_values = [value for sample in samples for value in sample.rollup.values("gpu_prefix_cache_hit_rate")]
    cpu_prefix_values = [value for sample in samples for value in sample.rollup.values("cpu_prefix_cache_hit_rate")]

    prefix = f"inference/{scope}"
    metrics: dict[str, float] = {
        f"{prefix}/running_requests": sum(running_values),
        f"{prefix}/waiting_requests": sum(waiting_values),
        f"{prefix}/running_imbalance": max_vio(running_values),
        f"{prefix}/waiting_imbalance": max_vio(waiting_values),
        f"{prefix}/nixl_failed_transfers_total": sum(
            sample.rollup.summed("nixl_failed_transfers_total") for sample in samples
        ),
        f"{prefix}/nixl_failed_notifications_total": sum(
            sample.rollup.summed("nixl_failed_notifications_total") for sample in samples
        ),
        f"{prefix}/nixl_kv_expired_requests_total": sum(
            sample.rollup.summed("nixl_kv_expired_requests_total") for sample in samples
        ),
    }

    if kv_values:
        metrics[f"{prefix}/kv_cache_usage_mean"] = sum(kv_values) / len(kv_values)
        metrics[f"{prefix}/kv_cache_usage_max"] = max(kv_values)
    if cpu_kv_values:
        metrics[f"{prefix}/cpu_kv_cache_usage_mean"] = sum(cpu_kv_values) / len(cpu_kv_values)
        metrics[f"{prefix}/cpu_kv_cache_usage_max"] = max(cpu_kv_values)

    prefix_cache_hit_rate = counter_ratio(samples, previous, "prefix_cache_hits", "prefix_cache_queries")
    if prefix_cache_hit_rate is None:
        prefix_cache_hit_rate = mean(prefix_gauge_values)
    if prefix_cache_hit_rate is not None:
        metrics[f"{prefix}/prefix_cache_hit_rate"] = prefix_cache_hit_rate
    cpu_prefix_cache_hit_rate = mean(cpu_prefix_values)
    if cpu_prefix_cache_hit_rate is not None:
        metrics[f"{prefix}/cpu_prefix_cache_hit_rate"] = cpu_prefix_cache_hit_rate

    prompt_token_rate = counter_rate(samples, previous, "prompt_tokens_total")
    generation_token_rate = counter_rate(samples, previous, "generation_tokens_total")
    if scope == "prefill":
        if prompt_token_rate is not None:
            metrics[f"{prefix}/throughput"] = prompt_token_rate
    elif scope == "decode":
        if generation_token_rate is not None:
            metrics[f"{prefix}/throughput"] = generation_token_rate
    else:
        token_rates = [rate for rate in (prompt_token_rate, generation_token_rate) if rate is not None]
        if token_rates:
            metrics[f"{prefix}/throughput"] = sum(token_rates)

    counter_metrics = {
        "completed_requests": "request_success_total",
        "nixl_transfers_per_second": "nixl_xfer_time_seconds_count",
        "nixl_bytes_per_second": "nixl_bytes_transferred_sum",
    }
    for metric_name, attribute in counter_metrics.items():
        rate = counter_rate(samples, previous, attribute)
        if rate is not None:
            metrics[f"{prefix}/{metric_name}"] = rate

    histogram_metrics = {
        "avg_queue_time_seconds": (
            "request_queue_time_seconds_sum",
            "request_queue_time_seconds_count",
        ),
        "avg_ttft_seconds": (
            "time_to_first_token_seconds_sum",
            "time_to_first_token_seconds_count",
        ),
        "avg_tpot_seconds": (
            "inter_token_latency_seconds_sum",
            "inter_token_latency_seconds_count",
        ),
        "avg_e2e_latency_seconds": (
            "e2e_request_latency_seconds_sum",
            "e2e_request_latency_seconds_count",
        ),
        "nixl_transfer_time_seconds": (
            "nixl_xfer_time_seconds_sum",
            "nixl_xfer_time_seconds_count",
        ),
        "nixl_avg_bytes_per_transfer": (
            "nixl_bytes_transferred_sum",
            "nixl_bytes_transferred_count",
        ),
    }
    for metric_name, (sum_attribute, count_attribute) in histogram_metrics.items():
        value = histogram_average(samples, previous, sum_attribute, count_attribute)
        if value is not None:
            metrics[f"{prefix}/{metric_name}"] = value

    prefill_time = histogram_average(
        samples,
        previous,
        "request_prefill_time_seconds_sum",
        "request_prefill_time_seconds_count",
    )
    decode_time = histogram_average(
        samples,
        previous,
        "request_decode_time_seconds_sum",
        "request_decode_time_seconds_count",
    )
    if scope == "prefill":
        if prefill_time is not None:
            metrics[f"{prefix}/avg_time_seconds"] = prefill_time
    elif scope == "decode":
        if decode_time is not None:
            metrics[f"{prefix}/avg_time_seconds"] = decode_time
    else:
        time_values = [value for value in (prefill_time, decode_time) if value is not None]
        if time_values:
            metrics[f"{prefix}/avg_time_seconds"] = sum(time_values) / len(time_values)

    return metrics


class InferenceMetricsCollector:
    """Polls vLLM Prometheus /metrics and logs smoothed role-aware values to W&B.

    The ``agg`` scope is always logged. The ``prefill`` and ``decode`` scopes are
    logged only when endpoints are explicitly or implicitly identified as a
    disaggregated P/D deployment.
    """

    def __init__(self, admin_clients: list[AsyncClient], roles: list[str | None] | None = None):
        self.endpoints = build_metrics_endpoints(admin_clients, roles=roles)
        self.metric_history: dict[str, deque[float]] = {}
        self.previous: dict[str, TimedRollup] = {}
        self.task: asyncio.Task | None = None
        self.has_pd_roles = {endpoint.role for endpoint in self.endpoints if endpoint.role is not None} == PD_ROLES

    async def start(self):
        wandb.define_metric("inference/*", step_metric="_timestamp")

        async def poll_loop():
            while True:
                try:
                    await self.collect_and_log()
                except Exception as e:
                    get_logger().debug(f"Inference metrics poll failed: {e!r}")
                await asyncio.sleep(POLL_INTERVAL)

        self.task = asyncio.create_task(poll_loop())

    async def collect_and_log(self):
        now = time.monotonic()

        async def fetch(endpoint: MetricsEndpoint) -> str | None:
            try:
                response = await endpoint.client.get("/metrics", timeout=5.0)
                response.raise_for_status()
                return response.text
            except Exception as e:
                get_logger().debug(f"Failed to fetch metrics from {endpoint.client.base_url}: {e!r}")
                return None

        results = await asyncio.gather(*[fetch(endpoint) for endpoint in self.endpoints])
        samples = [
            EndpointSample(endpoint=endpoint, timestamp=now, rollup=parse_prometheus_text(text))
            for endpoint, text in zip(self.endpoints, results)
            if text is not None
        ]
        if not samples:
            return

        metrics = build_scope_metrics("agg", samples, self.previous)
        if self.has_pd_roles:
            for role in sorted(PD_ROLES):
                role_samples = [sample for sample in samples if sample.endpoint.role == role]
                if role_samples:
                    metrics.update(build_scope_metrics(role, role_samples, self.previous))

        for sample in samples:
            self.previous[sample.endpoint.key] = TimedRollup(timestamp=sample.timestamp, rollup=sample.rollup)

        smoothed_metrics = self.smooth_metrics(metrics)
        if smoothed_metrics:
            smoothed_metrics["_timestamp"] = time.time()
            wandb.log(smoothed_metrics)

    def smooth_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Add current values to the smoothing window and return W&B-ready metrics."""
        for key, value in metrics.items():
            self.metric_history.setdefault(key, deque(maxlen=WINDOW_SIZE)).append(value)
        return {key: sum(values) / len(values) for key, values in self.metric_history.items() if values}

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
