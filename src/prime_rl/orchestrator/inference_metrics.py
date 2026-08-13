from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from statistics import mean, median

import wandb
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.logger import get_logger

POLL_INTERVAL = 5.0
FETCH_TIMEOUT = 5.0
METRIC_PREFIX = "vllm:"
PD_ROLES = {"prefill", "decode"}
QUANTILES = {"p50": 0.5, "p90": 0.9, "p99": 0.99}
AGGREGATIONS = {"min": min, "max": max, "sum": sum, "mean": mean, "median": median}
OVERLOAD_WAITING_FRACTION = 0.2
OVERLOAD_WARN_INTERVAL = 60.0

# Scope-level ratios derived from counter deltas; each operand lists legacy and
# OpenMetrics sample names, whichever the running vLLM version emits.
RATIO_METRICS = {
    "prefix_cache_hit_rate": (
        ("prefix_cache_hits", "prefix_cache_hits_total"),
        ("prefix_cache_queries", "prefix_cache_queries_total"),
    ),
}


@dataclass
class HistogramSnapshot:
    sum: float = 0.0
    count: float = 0.0
    buckets: dict[float, float] = field(default_factory=dict)
    """Upper bound (``le``) -> cumulative count."""


@dataclass
class EngineSnapshot:
    """All samples scraped for one engine, keyed by vLLM sample name.

    Samples that carry extra labels (e.g. ``finished_reason``) are summed over
    those labels so each name maps to a single value per engine.
    """

    gauges: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, HistogramSnapshot] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsEndpoint:
    client: AsyncClient
    role: str | None
    key: str
    name: str


@dataclass(frozen=True)
class TimedSnapshot:
    timestamp: float
    snapshot: EngineSnapshot


@dataclass(frozen=True)
class EngineSample:
    endpoint: MetricsEndpoint
    engine_label: str
    timestamp: float
    snapshot: EngineSnapshot

    @property
    def engine_id(self) -> str:
        return f"{self.endpoint.name}.{self.engine_label}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.endpoint.key, self.engine_label)


def parse_prometheus_text(text: str) -> dict[str, EngineSnapshot]:
    """Parse a vLLM Prometheus exposition into one snapshot per engine label."""
    engines: dict[str, EngineSnapshot] = {}
    for family in text_string_to_metric_families(text):
        if not family.name.startswith(METRIC_PREFIX) or family.type not in ("counter", "gauge", "histogram"):
            continue
        for sample in family.samples:
            if sample.name.endswith("_created") or not math.isfinite(sample.value):
                continue
            engine = engines.setdefault(sample.labels.get("engine", "0"), EngineSnapshot())
            name = sample.name.removeprefix(METRIC_PREFIX)
            if family.type == "gauge":
                engine.gauges[name] = engine.gauges.get(name, 0.0) + sample.value
            elif family.type == "counter":
                engine.counters[name] = engine.counters.get(name, 0.0) + sample.value
            else:
                histogram = engine.histograms.setdefault(family.name.removeprefix(METRIC_PREFIX), HistogramSnapshot())
                if sample.name.endswith("_sum"):
                    histogram.sum += sample.value
                elif sample.name.endswith("_count"):
                    histogram.count += sample.value
                elif sample.name.endswith("_bucket"):
                    le = float(sample.labels["le"])
                    histogram.buckets[le] = histogram.buckets.get(le, 0.0) + sample.value
    return engines


def build_metrics_endpoints(
    admin_clients: list[AsyncClient], roles: list[str | None] | None = None
) -> list[MetricsEndpoint]:
    """Attach optional P/D roles and stable display names to admin clients."""
    if roles is None:
        roles = [None] * len(admin_clients)
    if len(roles) != len(admin_clients):
        raise ValueError(f"Got {len(roles)} inference metric role(s) for {len(admin_clients)} admin client(s)")

    endpoints: list[MetricsEndpoint] = []
    counts: dict[str, int] = {}
    for client, role in zip(admin_clients, roles):
        if role is not None and role not in PD_ROLES:
            raise ValueError(f"Unsupported inference metrics role: {role}")
        prefix = role or "server"
        index = counts.get(prefix, 0)
        counts[prefix] = index + 1
        endpoints.append(
            MetricsEndpoint(
                client=client,
                role=role,
                key=str(client.base_url).rstrip("/"),
                name=f"{prefix}{index}",
            )
        )
    return endpoints


def histogram_quantile(buckets: dict[float, float], quantile: float) -> float | None:
    """Estimate a quantile from cumulative bucket deltas, Prometheus-style."""
    total = max(buckets.values(), default=0.0)
    if total <= 0:
        return None
    target = quantile * total
    previous_le, previous_cumulative = 0.0, 0.0
    for le, cumulative in sorted(buckets.items()):
        if cumulative >= target:
            if math.isinf(le) or cumulative == previous_cumulative:
                return previous_le
            return previous_le + (le - previous_le) * (target - previous_cumulative) / (
                cumulative - previous_cumulative
            )
        previous_le, previous_cumulative = le, cumulative
    return None


def engine_values(sample: EngineSample, previous: TimedSnapshot | None) -> dict[str, float]:
    """Pass-through values plus interval-derived rates and means for one engine.

    Counter rates use ``:rate`` (per second), histogram interval means use
    ``:mean``, and histogram completion rates use ``:rate``, mirroring how one
    would express them as PromQL over the poll interval.
    """
    values: dict[str, float] = dict(sample.snapshot.gauges)
    values.update(sample.snapshot.counters)
    for name, histogram in sample.snapshot.histograms.items():
        values[f"{name}_sum"] = histogram.sum
        values[f"{name}_count"] = histogram.count

    if previous is None:
        return values
    dt = sample.timestamp - previous.timestamp
    if dt <= 0:
        return values

    for name, value in sample.snapshot.counters.items():
        delta = value - previous.snapshot.counters.get(name, 0.0)
        if delta >= 0:
            values[f"{name}:rate"] = delta / dt
    for name, histogram in sample.snapshot.histograms.items():
        previous_histogram = previous.snapshot.histograms.get(name, HistogramSnapshot())
        sum_delta = histogram.sum - previous_histogram.sum
        count_delta = histogram.count - previous_histogram.count
        if count_delta >= 0:
            values[f"{name}:rate"] = count_delta / dt
        if count_delta > 0 and sum_delta >= 0:
            values[f"{name}:mean"] = sum_delta / count_delta
    # Per-engine ratios so the scope aggregations include min/max — the pooled
    # scope-level ratio alone hides a single engine's collapse (e.g. one engine
    # thrashing at a 5% prefix hit rate inside a healthy fleet average).
    for name, (numerator_names, denominator_names) in RATIO_METRICS.items():
        numerator = sum(
            sample.snapshot.counters.get(c, 0.0) - previous.snapshot.counters.get(c, 0.0) for c in numerator_names
        )
        denominator = sum(
            sample.snapshot.counters.get(c, 0.0) - previous.snapshot.counters.get(c, 0.0) for c in denominator_names
        )
        if numerator >= 0 and denominator > 0:
            values[name] = numerator / denominator
    return values


def bucket_deltas(sample: EngineSample, previous: TimedSnapshot | None) -> dict[str, dict[float, float]]:
    """Per-histogram cumulative bucket deltas since the previous scrape."""
    if previous is None:
        return {}
    deltas: dict[str, dict[float, float]] = {}
    for name, histogram in sample.snapshot.histograms.items():
        previous_buckets = previous.snapshot.histograms.get(name, HistogramSnapshot()).buckets
        delta = {le: count - previous_buckets.get(le, 0.0) for le, count in histogram.buckets.items()}
        if delta and all(count >= 0 for count in delta.values()):
            deltas[name] = delta
    return deltas


def build_scope_metrics(
    scope: str,
    values_per_engine: list[dict[str, float]],
    bucket_deltas_per_engine: list[dict[str, dict[float, float]]],
    counter_deltas: dict[str, float],
) -> dict[str, float]:
    """Aggregate per-engine values into ``inference/{scope}/{metric}/{agg}`` series."""
    prefix = f"inference/{scope}"
    metrics: dict[str, float] = {}

    names = {name for values in values_per_engine for name in values}
    for name in sorted(names):
        engine_values_for_name = [values[name] for values in values_per_engine if name in values]
        for aggregation, fn in AGGREGATIONS.items():
            metrics[f"{prefix}/{name}/{aggregation}"] = fn(engine_values_for_name)

    histogram_names = {name for deltas in bucket_deltas_per_engine for name in deltas}
    for name in sorted(histogram_names):
        pooled: dict[float, float] = {}
        for deltas in bucket_deltas_per_engine:
            for le, count in deltas.get(name, {}).items():
                pooled[le] = pooled.get(le, 0.0) + count
        for label, quantile in QUANTILES.items():
            value = histogram_quantile(pooled, quantile)
            if value is not None:
                metrics[f"{prefix}/{name}/{label}"] = value

    for name, (numerator_names, denominator_names) in RATIO_METRICS.items():
        numerator = sum(counter_deltas.get(candidate, 0.0) for candidate in numerator_names)
        denominator = sum(counter_deltas.get(candidate, 0.0) for candidate in denominator_names)
        if denominator > 0:
            metrics[f"{prefix}/{name}/pooled"] = numerator / denominator

    return metrics


class InferenceMetricsCollector:
    """Polls vLLM Prometheus /metrics and mirrors every engine metric to W&B.

    All ``vllm:*`` families are passed through dynamically (the ``vllm:``
    prefix is stripped): gauges and counters verbatim, histograms as
    ``_sum``/``_count`` plus derived ``:rate``/``:mean`` and scope-level
    quantiles. Keys are ``inference/{scope}/{metric}/{stat}``: per-engine
    series under ``inference/{engine_id}/{metric}`` and cross-engine
    aggregations under ``inference/agg/{metric}/{stat}`` (plus
    ``inference/prefill|decode/...`` for disaggregated deployments). Engine
    ids (``server0.0``, ``prefill0.1``, ...) never collide with the scope
    names.
    """

    def __init__(
        self,
        admin_clients: list[AsyncClient],
        roles: list[str | None] | None = None,
        max_inflight_episodes: int | None = None,
    ):
        self.endpoints = build_metrics_endpoints(admin_clients, roles=roles)
        self.previous: dict[tuple[str, str], TimedSnapshot] = {}
        self.task: asyncio.Task | None = None
        self.has_pd_roles = {endpoint.role for endpoint in self.endpoints if endpoint.role is not None} == PD_ROLES
        self.max_inflight_episodes = max_inflight_episodes
        self.last_overload_warning = float("-inf")
        get_logger().info(
            "Collecting inference metrics from "
            + ", ".join(f"{endpoint.name}={endpoint.key}" for endpoint in self.endpoints)
        )

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
                response = await endpoint.client.get("/metrics", timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                return response.text
            except Exception as e:
                get_logger().debug(f"Failed to fetch metrics from {endpoint.client.base_url}: {e!r}")
                return None

        results = await asyncio.gather(*[fetch(endpoint) for endpoint in self.endpoints])
        samples = [
            EngineSample(endpoint=endpoint, engine_label=engine_label, timestamp=now, snapshot=snapshot)
            for endpoint, text in zip(self.endpoints, results)
            if text is not None
            for engine_label, snapshot in sorted(parse_prometheus_text(text).items())
        ]
        if not samples:
            return

        metrics = self.build_metrics(samples)
        for sample in samples:
            self.previous[sample.key] = TimedSnapshot(timestamp=sample.timestamp, snapshot=sample.snapshot)

        waiting_requests = metrics.get("inference/agg/num_requests_waiting/sum", 0.0)
        if (
            self.max_inflight_episodes is not None
            and waiting_requests > OVERLOAD_WAITING_FRACTION * self.max_inflight_episodes
            and now - self.last_overload_warning >= OVERLOAD_WARN_INTERVAL
        ):
            self.last_overload_warning = now
            get_logger().warning(
                f"Inference is overloaded: {waiting_requests:.0f} waiting request(s), more than "
                f"{OVERLOAD_WAITING_FRACTION:.0%} of max_inflight_episodes={self.max_inflight_episodes} - "
                "training is slowed by queued requests. If intermittent, wait it out - if persistent, "
                "lower concurrency by decreasing max_inflight_episodes"
            )

        if metrics:
            metrics["_timestamp"] = time.time()
            wandb.log(metrics)

    def build_metrics(self, samples: list[EngineSample]) -> dict[str, float]:
        values_per_engine = [engine_values(sample, self.previous.get(sample.key)) for sample in samples]
        bucket_deltas_per_engine = [bucket_deltas(sample, self.previous.get(sample.key)) for sample in samples]

        metrics: dict[str, float] = {}
        for sample, values in zip(samples, values_per_engine):
            for name, value in values.items():
                metrics[f"inference/{sample.engine_id}/{name}"] = value

        scopes: list[tuple[str, list[int]]] = [("agg", list(range(len(samples))))]
        if self.has_pd_roles:
            for role in sorted(PD_ROLES):
                indices = [i for i, sample in enumerate(samples) if sample.endpoint.role == role]
                if indices:
                    scopes.append((role, indices))

        for scope, indices in scopes:
            counter_deltas: dict[str, float] = {}
            for i in indices:
                previous = self.previous.get(samples[i].key)
                if previous is None:
                    continue
                for name, value in samples[i].snapshot.counters.items():
                    delta = value - previous.snapshot.counters.get(name, 0.0)
                    if delta >= 0:
                        counter_deltas[name] = counter_deltas.get(name, 0.0) + delta
            metrics.update(
                build_scope_metrics(
                    scope,
                    [values_per_engine[i] for i in indices],
                    [bucket_deltas_per_engine[i] for i in indices],
                    counter_deltas,
                )
            )
        return metrics

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
