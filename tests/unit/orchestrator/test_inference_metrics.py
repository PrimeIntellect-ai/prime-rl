import asyncio

import pytest
from httpx import AsyncClient

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector


async def _close_clients(clients: list[AsyncClient]) -> None:
    await asyncio.gather(*(client.aclose() for client in clients))


def test_server_metrics_are_namespaced_and_track_up():
    clients = [
        AsyncClient(base_url="http://ltc-idc3-hgx8-h200-6:8100"),
        AsyncClient(base_url="http://ltc-idc3-hgx8-h200-11:8200"),
    ]
    try:
        collector = InferenceMetricsCollector(clients)
        active_servers = {collector._server_names[0]}

        collector._update_server_histories(
            collector._server_names[0],
            10.0,
            {
                "vllm:num_requests_running": 3.0,
                "vllm:gpu_cache_usage_perc": 0.5,
                "vllm:gpu_prefix_cache_hit_rate": 0.2,
            },
            {"vllm:prompt_tokens": 100.0},
            {"vllm:nixl_xfer_time_seconds": (1.0, 2.0)},
        )
        collector._update_server_histories(
            collector._server_names[0],
            15.0,
            {
                "vllm:num_requests_running": 5.0,
                "vllm:gpu_cache_usage_perc": 0.7,
                "vllm:gpu_prefix_cache_hit_rate": 0.4,
            },
            {"vllm:prompt_tokens": 150.0},
            {"vllm:nixl_xfer_time_seconds": (1.5, 3.0)},
        )

        server_0 = "server_00_ltc-idc3-hgx8-h200-6_8100"
        server_1 = "server_01_ltc-idc3-hgx8-h200-11_8200"
        metrics = collector._server_metrics(active_servers)
        up_metrics = collector._server_up_metrics(active_servers)

        assert metrics[f"inference/server/{server_0}/num_requests_running"] == 4.0
        assert metrics[f"inference/server/{server_0}/gpu_cache_usage_perc_max"] == 0.6
        assert metrics[f"inference/server/{server_0}/kv_cache_left_perc_min"] == pytest.approx(0.4)
        assert metrics[f"inference/server/{server_0}/gpu_prefix_cache_hit_rate_max"] == pytest.approx(0.3)
        assert metrics[f"inference/server/{server_0}/kv_cache_hit_rate_max"] == pytest.approx(0.3)
        assert metrics[f"inference/server/{server_0}/prefill_throughput_tps"] == 10.0
        assert metrics[f"inference/server/{server_0}/nixl_xfer_time_seconds_avg_ms"] == 500.0
        assert all(f"/{server_1}/" not in key for key in metrics)
        assert up_metrics[f"inference/server/{server_0}/up"] == 1.0
        assert up_metrics[f"inference/server/{server_1}/up"] == 0.0
    finally:
        asyncio.run(_close_clients(clients))
