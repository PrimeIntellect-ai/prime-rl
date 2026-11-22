"""Tests for the Pro harness LLM clients (retry / content-filter / budget)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from glyphbench.pro_harness.clients import (
    AzureResponsesClient,
    CompletionResult,
    LLMClient,
    OpenAIChatClient,
    _ContentFilterError,
    _TransientError,
    _UnsupportedParamError,
)
from glyphbench.pro_harness.config import ProConfig


class _ScriptedClient(LLMClient):
    """Drives the base-class retry orchestration with a scripted _attempt."""

    def __init__(self, cfg, script):
        super().__init__(cfg)
        self.script = list(script)
        self.calls: list[dict] = []

    def describe(self) -> str:
        return "scripted"

    def _attempt(self, system, user, *, drop_sampling):
        self.calls.append({"system": system, "drop_sampling": drop_sampling})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(system, drop_sampling)
        return item


def _cfg(**kw):
    base = dict(
        task_id="glyphbench/craftaxfull-v0",
        backend="openai",
        max_retries=3,
        content_filter_retries=2,
        retry_backoff=1.0,
    )
    base.update(kw)
    return ProConfig(**base)


def test_content_filter_then_success_softens(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = _ScriptedClient(
        _cfg(),
        [_ContentFilterError("blocked"), CompletionResult(text="<action>DO</action>")],
    )
    res = c.complete("SYS", "USER")
    assert res.ok
    assert res.softened
    assert res.content_filter_retries == 1
    # Second attempt got a softened system prompt.
    assert c.calls[1]["system"].startswith("NOTE:")


def test_refusal_text_triggers_soften(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = _ScriptedClient(
        _cfg(),
        [
            CompletionResult(text="I'm sorry, but I can't assist with that."),
            CompletionResult(text="<action>DO</action>"),
        ],
    )
    res = c.complete("SYS", "USER")
    assert res.softened and "<action>DO</action>" in res.text


def test_content_filter_exhausted_returns_blocked(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = _ScriptedClient(
        _cfg(content_filter_retries=1),
        [_ContentFilterError("b1"), _ContentFilterError("b2")],
    )
    res = c.complete("SYS", "USER")
    assert res.blocked and res.finish_reason == "content_filter" and not res.ok


def test_transient_retry_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = _ScriptedClient(
        _cfg(max_retries=2),
        [_TransientError("503"), _TransientError("timeout"),
         CompletionResult(text="<action>DO</action>")],
    )
    res = c.complete("SYS", "USER")
    assert res.ok and res.attempts == 3
    assert res.transient_retries == 2


def test_transient_exhausted_returns_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = _ScriptedClient(
        _cfg(max_retries=1),
        [_TransientError("503"), _TransientError("503")],
    )
    res = c.complete("SYS", "USER")
    assert not res.ok and res.finish_reason == "error"


def test_unsupported_param_drops_sampling(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def ok_if_dropped(system, drop_sampling):
        assert drop_sampling, "should have dropped sampling on retry"
        return CompletionResult(text="<action>DO</action>")

    c = _ScriptedClient(
        _cfg(),
        [_UnsupportedParamError("temperature unsupported"), ok_if_dropped],
    )
    res = c.complete("SYS", "USER")
    assert res.ok
    assert res.unsupported_param_retries == 1
    assert c.calls[0]["drop_sampling"] is False
    assert c.calls[1]["drop_sampling"] is True


# --- OpenAIChatClient: budget + param building, no network ----------------
class _FakeUsage:
    prompt_tokens = 11
    prompt_tokens_details = SimpleNamespace(cached_tokens=5)
    completion_tokens = 7
    completion_tokens_details = None


class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None
        self.model_extra = None


class _FakeChoice:
    def __init__(self, content, finish="stop"):
        self.message = _FakeMsg(content)
        self.finish_reason = finish


class _FakeResp:
    def __init__(self, content, finish="stop"):
        self.choices = [_FakeChoice(content, finish)]
        self.usage = _FakeUsage()


def _make_openai_client(monkeypatch, captured, **cfg_kw):
    cfg = _cfg(backend="openai", model="dummy", base_url="http://x/v1", **cfg_kw)
    client = OpenAIChatClient(cfg)

    def fake_create(**kwargs):
        captured.append(kwargs)
        return _FakeResp("<action>DO</action>")

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    return client


def test_no_output_cap_when_max_tokens_none(monkeypatch):
    captured: list[dict] = []
    client = _make_openai_client(monkeypatch, captured, max_output_tokens=None)
    res = client.complete("SYS", "USER")
    assert res.ok
    assert res.cached_input_tokens == 5
    assert "max_completion_tokens" not in captured[0]  # reasoning never truncated


def test_output_cap_passed_when_set(monkeypatch):
    captured: list[dict] = []
    client = _make_openai_client(monkeypatch, captured, max_output_tokens=12345)
    client.complete("SYS", "USER")
    assert captured[0]["max_completion_tokens"] == 12345


def test_qwen_thinking_chat_template_flag_is_forwarded(monkeypatch):
    captured: list[dict] = []
    client = _make_openai_client(
        monkeypatch,
        captured,
        top_k=0,
        enable_thinking=True,
    )
    client.complete("SYS", "USER")
    assert captured[0]["extra_body"] == {
        "top_k": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_unsupported_param_fallback_openai(monkeypatch):
    import openai

    cfg = _cfg(backend="openai", model="dummy", base_url="http://x/v1",
               temperature=0.7, reasoning_effort="high")
    client = OpenAIChatClient(cfg)
    calls: list[dict] = []

    import httpx

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            resp = httpx.Response(400, request=httpx.Request("POST", "http://x/v1"))
            raise openai.BadRequestError(
                "Unsupported parameter: temperature",
                response=resp,
                body={"error": {"message": "Unsupported parameter: temperature"}},
            )
        return _FakeResp("<action>DO</action>")

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    res = client.complete("SYS", "USER")
    assert res.ok
    # First call had sampling params; second dropped them.
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_azure_responses_preserves_multimodal_image_input(monkeypatch):
    cfg = _cfg(
        backend="azure",
        model="gpt-5.6-sol",
        azure_endpoint="https://example.test/openai/v1/responses",
        azure_api_key="secret",
        reasoning_effort="high",
    )
    client = AzureResponsesClient(cfg)
    captured: list[dict] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "<action>DO</action>"}
                            ],
                        }
                    ],
                    "usage": {},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    multimodal = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Choose an action."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                    "detail": "original",
                },
            ],
        }
    ]
    result = client.complete("SYS", multimodal)

    assert result.ok
    assert captured[0]["input"] == multimodal
    assert captured[0]["reasoning"] == {"effort": "high"}
    assert "max_output_tokens" not in captured[0]
