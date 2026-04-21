from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from prometheus_client.parser import text_string_to_metric_families


COUNTER_KEYS = {
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
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
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
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

    def kv_values(self) -> list[float]:
        return [engine.kv_cache_usage_perc for engine in self.engines.values()]

    def cpu_kv_values(self) -> list[float]:
        return [engine.cpu_cache_usage_perc for engine in self.engines.values() if engine.cpu_cache_usage_perc is not None]

    def cpu_prefix_cache_hit_rate_values(self) -> list[float]:
        return [
            engine.cpu_prefix_cache_hit_rate
            for engine in self.engines.values()
            if engine.cpu_prefix_cache_hit_rate is not None
        ]


@dataclass
class RateSnapshot:
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    requests_finished_per_second: float = 0.0
    prefix_cache_hit_rate: float | None = None
    cpu_prefix_cache_hit_rate: float | None = None
    avg_prefill_time_seconds: float | None = None
    avg_decode_time_seconds: float | None = None
    avg_queue_time_seconds: float | None = None
    avg_ttft_seconds: float | None = None
    avg_tpot_seconds: float | None = None
    avg_e2e_latency_seconds: float | None = None
    nixl_avg_transfer_time_seconds: float | None = None
    nixl_transfers_per_second: float = 0.0
    nixl_bytes_per_second: float = 0.0
    nixl_avg_bytes_per_transfer: float | None = None


def parse_prometheus_text(payload: str) -> NodeRollup:
    engines: dict[str, EngineRollup] = defaultdict(EngineRollup)
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            engine_id = sample.labels.get("engine", "aggregate")
            engine = engines[engine_id]
            if sample.name in GAUGE_KEYS:
                setattr(engine, GAUGE_KEYS[sample.name], float(sample.value))
            elif sample.name in COUNTER_KEYS:
                if sample.name == "vllm:request_success_total":
                    engine.request_success_total += float(sample.value)
                else:
                    setattr(engine, COUNTER_KEYS[sample.name], float(sample.value))
            elif sample.name in HISTOGRAM_SUM_KEYS:
                setattr(engine, HISTOGRAM_SUM_KEYS[sample.name], float(sample.value))
            elif sample.name in HISTOGRAM_COUNT_KEYS:
                setattr(engine, HISTOGRAM_COUNT_KEYS[sample.name], float(sample.value))
    return NodeRollup(engines=dict(engines))


def _counter_rate(current: float, previous: float, dt: float) -> float:
    delta = current - previous
    if dt <= 0 or delta < 0:
        return 0.0
    return delta / dt


def _histogram_average(
    current_sum: float, current_count: float, previous_sum: float, previous_count: float
) -> float | None:
    sum_delta = current_sum - previous_sum
    count_delta = current_count - previous_count
    if sum_delta < 0 or count_delta <= 0:
        return None
    return sum_delta / count_delta


def _counter_ratio(
    current_numerator: float,
    current_denominator: float,
    previous_numerator: float,
    previous_denominator: float,
) -> float | None:
    numerator_delta = current_numerator - previous_numerator
    denominator_delta = current_denominator - previous_denominator
    if numerator_delta < 0 or denominator_delta <= 0:
        return None
    return numerator_delta / denominator_delta


def compute_rates(current: NodeRollup, previous: NodeRollup | None, dt: float) -> RateSnapshot:
    if previous is None:
        cpu_prefix_values = current.cpu_prefix_cache_hit_rate_values()
        return RateSnapshot(
            cpu_prefix_cache_hit_rate=(
                sum(cpu_prefix_values) / len(cpu_prefix_values) if cpu_prefix_values else None
            )
        )

    cpu_prefix_values = current.cpu_prefix_cache_hit_rate_values()
    return RateSnapshot(
        prompt_tokens_per_second=_counter_rate(
            current.summed("prompt_tokens_total"),
            previous.summed("prompt_tokens_total"),
            dt,
        ),
        generation_tokens_per_second=_counter_rate(
            current.summed("generation_tokens_total"),
            previous.summed("generation_tokens_total"),
            dt,
        ),
        requests_finished_per_second=_counter_rate(
            current.summed("request_success_total"),
            previous.summed("request_success_total"),
            dt,
        ),
        prefix_cache_hit_rate=_counter_ratio(
            current.summed("prefix_cache_hits"),
            current.summed("prefix_cache_queries"),
            previous.summed("prefix_cache_hits"),
            previous.summed("prefix_cache_queries"),
        ),
        cpu_prefix_cache_hit_rate=(
            sum(cpu_prefix_values) / len(cpu_prefix_values) if cpu_prefix_values else None
        ),
        avg_prefill_time_seconds=_histogram_average(
            current.summed("request_prefill_time_seconds_sum"),
            current.summed("request_prefill_time_seconds_count"),
            previous.summed("request_prefill_time_seconds_sum"),
            previous.summed("request_prefill_time_seconds_count"),
        ),
        avg_decode_time_seconds=_histogram_average(
            current.summed("request_decode_time_seconds_sum"),
            current.summed("request_decode_time_seconds_count"),
            previous.summed("request_decode_time_seconds_sum"),
            previous.summed("request_decode_time_seconds_count"),
        ),
        avg_queue_time_seconds=_histogram_average(
            current.summed("request_queue_time_seconds_sum"),
            current.summed("request_queue_time_seconds_count"),
            previous.summed("request_queue_time_seconds_sum"),
            previous.summed("request_queue_time_seconds_count"),
        ),
        avg_ttft_seconds=_histogram_average(
            current.summed("time_to_first_token_seconds_sum"),
            current.summed("time_to_first_token_seconds_count"),
            previous.summed("time_to_first_token_seconds_sum"),
            previous.summed("time_to_first_token_seconds_count"),
        ),
        avg_tpot_seconds=_histogram_average(
            current.summed("inter_token_latency_seconds_sum"),
            current.summed("inter_token_latency_seconds_count"),
            previous.summed("inter_token_latency_seconds_sum"),
            previous.summed("inter_token_latency_seconds_count"),
        ),
        avg_e2e_latency_seconds=_histogram_average(
            current.summed("e2e_request_latency_seconds_sum"),
            current.summed("e2e_request_latency_seconds_count"),
            previous.summed("e2e_request_latency_seconds_sum"),
            previous.summed("e2e_request_latency_seconds_count"),
        ),
        nixl_avg_transfer_time_seconds=_histogram_average(
            current.summed("nixl_xfer_time_seconds_sum"),
            current.summed("nixl_xfer_time_seconds_count"),
            previous.summed("nixl_xfer_time_seconds_sum"),
            previous.summed("nixl_xfer_time_seconds_count"),
        ),
        nixl_transfers_per_second=_counter_rate(
            current.summed("nixl_xfer_time_seconds_count"),
            previous.summed("nixl_xfer_time_seconds_count"),
            dt,
        ),
        nixl_bytes_per_second=_counter_rate(
            current.summed("nixl_bytes_transferred_sum"),
            previous.summed("nixl_bytes_transferred_sum"),
            dt,
        ),
        nixl_avg_bytes_per_transfer=_histogram_average(
            current.summed("nixl_bytes_transferred_sum"),
            current.summed("nixl_bytes_transferred_count"),
            previous.summed("nixl_bytes_transferred_sum"),
            previous.summed("nixl_bytes_transferred_count"),
        ),
    )
