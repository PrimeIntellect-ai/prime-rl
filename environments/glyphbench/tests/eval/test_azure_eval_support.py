from __future__ import annotations

import importlib.util
import io
import json
import threading
import urllib.error
from pathlib import Path

import pytest

from glyphbench.azure_openai import azure_auth_headers, normalize_responses_endpoint
from glyphbench.eval_tracking import aggregate_episodes, load_normal_episodes


def _load_proxy_module():
    path = Path(__file__).parents[2] / "scripts" / "azure_responses_proxy.py"
    spec = importlib.util.spec_from_file_location("glyphbench_test_azure_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_foundry_endpoint_normalization_and_bearer_auth() -> None:
    base = "https://Example.services.ai.azure.com/openai/v1"
    endpoint = normalize_responses_endpoint(base)
    assert endpoint == f"{base}/responses"
    assert azure_auth_headers(endpoint, "secret") == {"Authorization": "Bearer secret"}


def test_legacy_endpoint_uses_api_key_auth() -> None:
    endpoint = "https://example.openai.azure.com/openai/responses?api-version=preview"
    assert normalize_responses_endpoint(endpoint) == endpoint
    assert azure_auth_headers(endpoint, "secret") == {"api-key": "secret"}


def test_proxy_classifies_policy_block_wrapped_in_generic_error() -> None:
    proxy = _load_proxy_module()
    body = json.dumps(
        {
            "error": {
                "code": "invalid_request_error",
                "type": "invalid_request_error",
                "message": "Request blocked by the ResponsibleAI content management policy",
            }
        }
    )
    assert proxy._azure_error_code(body) == "content_filter"


def test_proxy_uncapped_payload_and_degraded_fallback(monkeypatch) -> None:
    proxy = _load_proxy_module()
    monkeypatch.setenv("AZURE_OPENAI_REASONING_EFFORT", "high")
    body = {
        "model": "deployment",
        "messages": [
            {"role": "system", "content": "play"},
            {"role": "user", "content": "choose a move"},
        ],
    }
    payload = proxy._build_azure_payload(body, "fallback")
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert payload["reasoning"] == {"effort": "high"}
    degraded = proxy._degraded_chat_response(body, "deployment", "content_filter")
    text = degraded["choices"][0]["message"]["content"]
    assert "<action>NOOP</action>" in text
    assert degraded["glyphbench_proxy"]["degraded"] is True


def test_proxy_honors_request_reasoning_and_preserves_usage(monkeypatch) -> None:
    proxy = _load_proxy_module()
    monkeypatch.setenv("AZURE_OPENAI_REASONING_EFFORT", "high")
    payload = proxy._build_azure_payload(
        {"messages": [], "reasoning_effort": "none"}, "model"
    )
    assert payload["reasoning"] == {"effort": "none"}
    result = proxy._chat_response_from_azure(
        {
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "private reasoning"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "<action>NOOP</action>"}]},
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 80},
                "output_tokens": 40,
                "output_tokens_details": {"reasoning_tokens": 30},
            },
        },
        "model",
    )
    assert result["choices"][0]["message"]["content"] == "<action>NOOP</action>"
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 30


@pytest.mark.parametrize("failure", ["rate_limit", "connection"])
def test_proxy_retries_transient_failures(monkeypatch, failure) -> None:
    proxy = _load_proxy_module()
    handler = object.__new__(proxy.AzureResponsesProxy)
    handler.path = "/v1/chat/completions"
    body = json.dumps({"messages": [{"role": "user", "content": "play"}]}).encode()
    handler.headers = {"content-length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.endpoint = "https://example.test/responses"
    handler.api_key = "test-key"
    handler.auth_mode = "api-key"
    handler.policy_id = None
    handler.model = "test-model"
    handler.timeout = 1
    handler.invalid_prompt_retries = handler.content_filter_retries = handler.model_error_retries = 0
    handler.transient_retries = 1
    handler.invalid_prompt_retry_delay = 0
    proxy.AzureResponsesProxy.metrics = {}
    proxy.AzureResponsesProxy.metrics_lock = threading.Lock()
    results = []
    handler._send_json = lambda status, value: results.append((status, value))
    attempts = 0

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "connection":
                raise urllib.error.URLError("connection reset")
            raise urllib.error.HTTPError(
                handler.endpoint, 429, "rate limit", {},
                io.BytesIO(b'{"error":{"code":"rate_limit_exceeded"}}'),
            )
        return io.BytesIO(b'{"status":"completed","output_text":"<action>NOOP</action>"}')

    monkeypatch.setattr(proxy.urllib.request, "urlopen", urlopen)
    handler.do_POST()
    assert attempts == 2
    assert results[0][0] == 200
    assert results[0][1]["choices"][0]["message"]["content"] == "<action>NOOP</action>"


def test_normal_tracking_reads_return_achievement_names_and_gif(tmp_path: Path) -> None:
    run = tmp_path / "normal-run"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"env_id": "glyphbench/craftaxfull-v0"}), encoding="utf-8"
    )
    gif = run / "native.gif"
    gif.write_bytes(b"GIF89a")
    row = {
        "reward": 0.75,
        "metrics": {"forfeit_rate": 0.0, "parse_recovery_rate": 0.1},
        "token_usage": {"input_tokens": 10, "output_tokens": 5},
        "info": {
            "env_id": "glyphbench/craftaxfull-v0",
            "seed": 42,
            "gif_path": str(gif),
            "last_env_info": {"achievements": ["collect_wood", "place_table"]},
        },
    }
    (run / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    episodes = load_normal_episodes(tmp_path)
    assert episodes[0]["raw_score"] == 2.0
    assert episodes[0]["achievement_names"] == ["collect_wood", "place_table"]
    assert episodes[0]["gif_path"] == str(gif)
    summary = aggregate_episodes(episodes)
    assert summary["eval/return_mean"] == 0.75
    assert summary["eval/achievements_mean"] == 2.0
