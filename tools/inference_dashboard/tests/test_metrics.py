from inference_dashboard.metrics import compute_rates, parse_prometheus_text


PROMETHEUS_PAYLOAD = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 3
vllm:num_requests_running{engine="1"} 2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 5
vllm:num_requests_waiting{engine="1"} 7
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.50
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc{engine="0"} 0.10
vllm:cpu_cache_usage_perc{engine="1"} 0.20
# TYPE vllm:cpu_prefix_cache_hit_rate gauge
vllm:cpu_prefix_cache_hit_rate{engine="0"} 0.60
vllm:cpu_prefix_cache_hit_rate{engine="1"} 0.80
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0"} 100
vllm:prompt_tokens_total{engine="1"} 120
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0"} 40
vllm:generation_tokens_total{engine="1"} 80
# TYPE vllm:request_success counter
vllm:request_success_total{engine="0",finished_reason="stop"} 10
vllm:request_success_total{engine="1",finished_reason="stop"} 12
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0"} 40
vllm:prefix_cache_queries{engine="1"} 60
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="0"} 30
vllm:prefix_cache_hits{engine="1"} 24
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_sum{engine="0"} 4
vllm:request_prefill_time_seconds_count{engine="0"} 2
vllm:request_prefill_time_seconds_sum{engine="1"} 6
vllm:request_prefill_time_seconds_count{engine="1"} 3
# TYPE vllm:nixl_xfer_time_seconds histogram
vllm:nixl_xfer_time_seconds_sum{engine="0"} 2
vllm:nixl_xfer_time_seconds_count{engine="0"} 4
vllm:nixl_xfer_time_seconds_sum{engine="1"} 3
vllm:nixl_xfer_time_seconds_count{engine="1"} 6
# TYPE vllm:nixl_bytes_transferred histogram
vllm:nixl_bytes_transferred_sum{engine="0"} 1000
vllm:nixl_bytes_transferred_count{engine="0"} 4
vllm:nixl_bytes_transferred_sum{engine="1"} 1800
vllm:nixl_bytes_transferred_count{engine="1"} 6
"""


def test_parse_prometheus_text_aggregates_engines():
    rollup = parse_prometheus_text(PROMETHEUS_PAYLOAD)

    assert rollup.engine_count == 2
    assert rollup.summed("running_requests") == 5
    assert rollup.summed("waiting_requests") == 12
    assert rollup.kv_values() == [0.25, 0.5]
    assert rollup.cpu_kv_values() == [0.1, 0.2]
    assert rollup.cpu_prefix_cache_hit_rate_values() == [0.6, 0.8]
    assert rollup.summed("request_success_total") == 22
    assert rollup.summed("prefix_cache_queries") == 100
    assert rollup.summed("prefix_cache_hits") == 54


def test_compute_rates_uses_counter_and_histogram_deltas():
    previous = parse_prometheus_text(PROMETHEUS_PAYLOAD)
    current = parse_prometheus_text(
        PROMETHEUS_PAYLOAD.replace('vllm:prompt_tokens_total{engine="0"} 100', 'vllm:prompt_tokens_total{engine="0"} 160')
        .replace('vllm:prompt_tokens_total{engine="1"} 120', 'vllm:prompt_tokens_total{engine="1"} 200')
        .replace('vllm:generation_tokens_total{engine="0"} 40', 'vllm:generation_tokens_total{engine="0"} 90')
        .replace('vllm:generation_tokens_total{engine="1"} 80', 'vllm:generation_tokens_total{engine="1"} 130')
        .replace('vllm:request_success_total{engine="0",finished_reason="stop"} 10', 'vllm:request_success_total{engine="0",finished_reason="stop"} 16')
        .replace('vllm:request_success_total{engine="1",finished_reason="stop"} 12', 'vllm:request_success_total{engine="1",finished_reason="stop"} 20')
        .replace('vllm:prefix_cache_queries{engine="0"} 40', 'vllm:prefix_cache_queries{engine="0"} 70')
        .replace('vllm:prefix_cache_queries{engine="1"} 60', 'vllm:prefix_cache_queries{engine="1"} 90')
        .replace('vllm:prefix_cache_hits{engine="0"} 30', 'vllm:prefix_cache_hits{engine="0"} 52')
        .replace('vllm:prefix_cache_hits{engine="1"} 24', 'vllm:prefix_cache_hits{engine="1"} 39')
        .replace(
            '4\nvllm:request_prefill_time_seconds_count{engine="0"} 2',
            '10\nvllm:request_prefill_time_seconds_count{engine="0"} 5',
        )
        .replace(
            '6\nvllm:request_prefill_time_seconds_count{engine="1"} 3',
            '12\nvllm:request_prefill_time_seconds_count{engine="1"} 6',
        )
        .replace(
            '2\nvllm:nixl_xfer_time_seconds_count{engine="0"} 4', '6\nvllm:nixl_xfer_time_seconds_count{engine="0"} 8'
        )
        .replace(
            '3\nvllm:nixl_xfer_time_seconds_count{engine="1"} 6', '7\nvllm:nixl_xfer_time_seconds_count{engine="1"} 10'
        )
        .replace(
            '1000\nvllm:nixl_bytes_transferred_count{engine="0"} 4',
            '1800\nvllm:nixl_bytes_transferred_count{engine="0"} 8',
        )
        .replace(
            '1800\nvllm:nixl_bytes_transferred_count{engine="1"} 6',
            '3000\nvllm:nixl_bytes_transferred_count{engine="1"} 10',
        )
    )

    rates = compute_rates(current, previous, 4.0)

    assert rates.prompt_tokens_per_second == 35.0
    assert rates.generation_tokens_per_second == 25.0
    assert rates.requests_finished_per_second == 3.5
    assert rates.prefix_cache_hit_rate == 37 / 60
    assert rates.cpu_prefix_cache_hit_rate == 0.7
    assert rates.avg_prefill_time_seconds == 2.0
    assert rates.nixl_transfers_per_second == 2.0
    assert rates.nixl_avg_transfer_time_seconds == 1.0
    assert rates.nixl_avg_bytes_per_transfer == 250.0
