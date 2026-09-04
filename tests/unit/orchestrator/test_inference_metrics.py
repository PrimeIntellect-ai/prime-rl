import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.orchestrator.inference_metrics import (
    MAX_CONCURRENT_METRICS_FETCHES,
    MAX_METRICS_RESPONSE_BYTES,
    InferenceMetricsCollector,
    parse_bounded_prometheus_text,
)


def response(*, text: str = "", payload: dict | None = None) -> MagicMock:
    result = MagicMock()
    result.text = text
    result.json.return_value = payload or {}
    result.raise_for_status.return_value = None
    return result


def test_metrics_collector_uses_bounded_streaming_requests():
    client = MagicMock()
    client.base_url = "http://worker:8120"
    collector = InferenceMetricsCollector([client], log_metrics=False)
    metrics = response(text="# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n")

    with patch(
        "prime_rl.orchestrator.inference_metrics._bounded_request",
        new=AsyncMock(side_effect=[metrics, response(payload={"data": []})]),
    ) as request:
        asyncio.run(collector.collect_and_log())

    metrics_call, models_call = request.await_args_list
    assert metrics_call.args == (client, "GET", "/metrics")
    assert metrics_call.kwargs["max_response_bytes"] == MAX_METRICS_RESPONSE_BYTES
    assert models_call.args == (client, "GET", "/v1/models")
    assert "max_response_bytes" not in models_call.kwargs


def test_oversized_metrics_scrape_does_not_update_snapshot():
    client = MagicMock()
    client.base_url = "http://worker:8120"
    collector = InferenceMetricsCollector([client], log_metrics=False)

    with patch(
        "prime_rl.orchestrator.inference_metrics._bounded_request",
        new=AsyncMock(side_effect=ValueError("Admin response body exceeds limit")),
    ):
        asyncio.run(collector.collect_and_log())

    assert collector.previous == {}


def test_metrics_collector_bounds_fleet_wide_response_concurrency():
    clients = []
    for index in range(MAX_CONCURRENT_METRICS_FETCHES + 2):
        client = MagicMock()
        client.base_url = f"http://worker-{index}:8120"
        clients.append(client)
    collector = InferenceMetricsCollector(clients, log_metrics=False)
    active = 0
    peak = 0

    async def bounded_response(client, method, path, **kwargs):
        nonlocal active, peak
        if path == "/v1/models":
            return response(payload={"data": []})
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response(text="# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n")

    with patch(
        "prime_rl.orchestrator.inference_metrics._bounded_request",
        new=AsyncMock(side_effect=bounded_response),
    ):
        asyncio.run(collector.collect_and_log())

    assert peak == MAX_CONCURRENT_METRICS_FETCHES


def test_metrics_parser_rejects_excessive_line_cardinality(monkeypatch):
    monkeypatch.setattr("prime_rl.orchestrator.inference_metrics.MAX_METRICS_LINES", 2)

    with pytest.raises(ValueError, match="exceeds 2 lines"):
        parse_bounded_prometheus_text("# first\n# second\n# third\n")
