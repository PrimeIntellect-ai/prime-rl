"""Real-HTTP integration tests: the Pro harness clients against fake OpenAI
chat-completions and Azure Responses servers, exercising the content-filter
soften-retry path and the no-output-cap (no reasoning truncation) contract."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from glyphbench.pro_harness import ProConfig, ProHarness


# --- shared recording state ------------------------------------------------
class _Recorder:
    def __init__(self):
        self.bodies: list[dict] = []
        self.first_filtered = False  # have we already returned one filter error?


def _action_or_memory_text(body_text: str) -> str:
    if "[Memory Update]" in body_text or "MEMORY TURN" in body_text:
        return ("<memory>\nTACTICAL: exploring.\n"
                "MAP_ADD: area=0, type=water, row=1, col=1, note=pool\n</memory>")
    return "Reasoning about the grid.\n<action>MOVE_RIGHT</action>"


def _make_chat_handler(rec: _Recorder, filter_first: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(n)
            body = json.loads(raw or b"{}")
            rec.bodies.append(body)
            if filter_first and not rec.first_filtered:
                rec.first_filtered = True
                self._send(400, {"error": {
                    "code": "content_filter", "type": "invalid_request_error",
                    "message": "The response was filtered by content management policy.",
                }})
                return
            user_text = body["messages"][-1]["content"]
            content = _action_or_memory_text(user_text)
            self._send(200, {
                "id": "x", "object": "chat.completion", "created": 0,
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            })

    return Handler


def _make_responses_handler(rec: _Recorder, filter_first: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            rec.bodies.append(body)
            if filter_first and not rec.first_filtered:
                rec.first_filtered = True
                self._send(400, {"error": {
                    "code": "content_filter",
                    "message": "blocked by responsible AI policy",
                }})
                return
            user_text = body.get("input", "")
            if isinstance(user_text, list):
                user_text = " ".join(str(p.get("content", "")) for p in user_text)
            content = _action_or_memory_text(user_text)
            self._send(200, {
                "id": "resp-x", "status": "completed", "model": body.get("model"),
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": content}]}],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            })

    return Handler


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_openai_chat_backend_end_to_end():
    rec = _Recorder()
    srv = _serve(_make_chat_handler(rec, filter_first=True))
    try:
        port = srv.server_address[1]
        cfg = ProConfig(
            task_id="glyphbench/craftaxfull-v0", backend="openai",
            base_url=f"http://127.0.0.1:{port}/v1", model="fake", api_key="EMPTY",
            max_turns=3, max_output_tokens=None, save_results=False, verbose=False,
        )
        out = ProHarness(cfg).run()
    finally:
        srv.shutdown()

    assert out["summary"]["episodes"] == 1
    # The first action call was content-filtered then softened and recovered.
    assert out["summary"]["total_softened_calls"] >= 1
    assert out["summary"]["total_forfeits"] == 0
    # No output-token cap was ever sent (no reasoning truncation).
    assert all("max_completion_tokens" not in b for b in rec.bodies)
    # The softened retry prepended the benign preamble to the system prompt.
    assert any(b["messages"][0]["content"].startswith("NOTE:") for b in rec.bodies)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_azure_responses_backend_end_to_end():
    rec = _Recorder()
    srv = _serve(_make_responses_handler(rec, filter_first=True))
    try:
        port = srv.server_address[1]
        endpoint = f"http://127.0.0.1:{port}/responses"
        cfg = ProConfig(
            task_id="glyphbench/craftaxfull-v0", backend="azure",
            azure_endpoint=endpoint, azure_api_key="k", model="gpt-5.4",
            reasoning_effort="high", max_turns=3, max_output_tokens=None,
            save_results=False, verbose=False,
        )
        out = ProHarness(cfg).run()
    finally:
        srv.shutdown()

    assert out["summary"]["episodes"] == 1
    assert out["summary"]["total_softened_calls"] >= 1
    assert out["summary"]["total_forfeits"] == 0
    # reasoning.effort forwarded; no max_output_tokens cap sent.
    assert any(b.get("reasoning", {}).get("effort") == "high" for b in rec.bodies)
    assert all("max_output_tokens" not in b for b in rec.bodies)
    # Softened retry reworded the instructions (system) field.
    assert any(str(b.get("instructions", "")).startswith("NOTE:") for b in rec.bodies)
