import math

from prime_rl.orchestrator.inference_metrics import parse_engine_metrics


def test_parse_engine_metrics():
    payload = {
        "router_queued": 3,
        "workers": [
            {
                "url": "http://prefill:8100",
                "role": "prefill",
                "healthy": True,
                "router_inflight": 2,
                "max_model_len": 32768,
                "engines": {
                    "0": {
                        "gauges": {"num_requests_waiting": 4.0, "num_requests_waiting_reason_capacity": 1.0},
                        "counters": {"request_success_total": 10.0},
                        "histograms": {
                            "e2e_request_latency_seconds": {
                                "sum": 42.5,
                                "count": 14.0,
                                "buckets": {"1.0": 3.0, "+Inf": 14.0},
                            }
                        },
                        "cache_config": {"num_gpu_blocks": "1000", "block_size": "16"},
                    }
                },
            },
            {
                "url": "http://engine:8100",
                "role": "regular",
                "healthy": False,
                "router_inflight": 0,
                "engines": {},
                "error": "connection refused",
            },
        ],
    }

    router_queued, (prefill, regular) = parse_engine_metrics(payload)

    assert router_queued == 3
    assert prefill.role == "prefill"
    assert prefill.max_model_len == 32768
    snapshot = prefill.engines["0"]
    assert snapshot.gauges["num_requests_waiting_reason_capacity"] == 1.0
    assert snapshot.counters["request_success_total"] == 10.0
    assert snapshot.histograms["e2e_request_latency_seconds"].buckets == {1.0: 3.0, math.inf: 14.0}
    assert snapshot.cache_config["num_gpu_blocks"] == "1000"
    assert regular.role is None
    assert regular.max_model_len is None
    assert regular.error == "connection refused"
