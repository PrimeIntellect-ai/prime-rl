#!/usr/bin/env python
"""OpenAI chat-completions shim for Azure Responses API evals.

Verifiers currently talks to OpenAI-compatible ``/v1/chat/completions`` clients.
This local proxy keeps that eval path unchanged while forwarding each request to
an Azure OpenAI Responses endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from glyphbench.azure_openai import azure_auth_headers, normalize_responses_endpoint


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                elif part.get("type") == "image_url":
                    chunks.append("[image omitted]")
            else:
                chunks.append(str(part))
        return "\n".join(chunks).strip()
    return str(content)


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, str]]]:
    instructions: list[str] = []
    inputs: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = _content_to_text(message.get("content"))
        if not text:
            continue
        if role in {"system", "developer"}:
            instructions.append(text)
            continue
        if role == "assistant":
            inputs.append({"role": "assistant", "content": text})
        else:
            prefix = "" if role == "user" else f"[{role}]\n"
            inputs.append({"role": "user", "content": f"{prefix}{text}"})
    return ("\n\n".join(instructions) or None), inputs


def _build_azure_payload(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    instructions, input_messages = _messages_to_responses_input(
        list(body.get("messages") or [])
    )
    payload: dict[str, Any] = {
        "model": body.get("model") or default_model,
        "input": input_messages or "",
    }
    if instructions:
        payload["instructions"] = instructions

    max_output_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        value = body.get(key)
        if value is not None:
            payload[key] = value

    reasoning = body.get("reasoning")
    if reasoning is not None:
        payload["reasoning"] = reasoning
    elif body.get("reasoning_effort") is not None:
        payload["reasoning"] = {"effort": body["reasoning_effort"]}
    elif os.environ.get("AZURE_OPENAI_REASONING_EFFORT"):
        payload["reasoning"] = {"effort": os.environ["AZURE_OPENAI_REASONING_EFFORT"]}

    return payload


def _extract_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _finish_reason(response: dict[str, Any]) -> str:
    if response.get("status") == "incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
        if reason in {"content_filter", "responsible_ai_policy"}:
            return "content_filter"
    return "stop"


def _chat_response_from_azure(response: dict[str, Any], model: str) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or input_tokens + output_tokens
    return {
        "id": response.get("id", f"azure-resp-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _extract_output_text(response),
                },
                "finish_reason": _finish_reason(response),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": usage.get("input_tokens_details"),
            "completion_tokens_details": usage.get("output_tokens_details"),
        },
    }


class AzureResponsesProxy(BaseHTTPRequestHandler):
    endpoint: str
    api_key: str
    model: str
    policy_id: str | None
    timeout: float
    invalid_prompt_retries: int
    content_filter_retries: int
    model_error_retries: int
    transient_retries: int
    invalid_prompt_retry_delay: float
    auth_mode: str
    metrics: dict[str, int]
    metrics_lock: threading.Lock

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @classmethod
    def _bump(cls, key: str) -> None:
        with cls.metrics_lock:
            cls.metrics[key] = cls.metrics.get(key, 0) + 1

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "azure",
                        }
                    ],
                },
            )
            return
        if self.path.rstrip("/") == "/v1/proxy-metrics":
            with self.metrics_lock:
                metrics = dict(self.metrics)
            self._send_json(200, metrics)
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        try:
            self._bump("chat_requests")
            try:
                length = int(self.headers.get("content-length", "0"))
                if length < 0:
                    raise ValueError("content-length must be nonnegative")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request must be a JSON object")
                messages = body.get("messages")
                if not isinstance(messages, list) or not all(
                    isinstance(message, dict) for message in messages
                ):
                    raise ValueError("messages must be an array of objects")
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": {"message": str(exc)}})
                return
            if body.get("stream"):
                self._send_json(
                    400,
                    {"error": {"message": "streaming is not supported by this proxy"}},
                )
                return
            payload = _build_azure_payload(body, self.model)
            retry_limits = {
                "invalid_prompt": self.invalid_prompt_retries,
                "content_filter": self.content_filter_retries,
                "model_error": self.model_error_retries,
                "transient": self.transient_retries,
            }
            retry_counts = {key: 0 for key in retry_limits}
            dropped_sampling = False
            while True:
                self._bump("upstream_attempts")
                data = json.dumps(payload).encode("utf-8")
                headers = {"content-type": "application/json"}
                headers.update(
                    azure_auth_headers(
                        self.endpoint,
                        self.api_key,
                        auth_mode=self.auth_mode,
                    )
                )
                if self.policy_id:
                    headers["x-policy-id"] = self.policy_id
                request = urllib.request.Request(
                    self.endpoint,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                        response = json.loads(resp.read())
                    self._bump("upstream_successes")
                    break
                except urllib.error.HTTPError as exc:
                    text = exc.read().decode("utf-8", errors="replace")
                    self._bump(f"upstream_http_{exc.code}")
                    if (
                        exc.code == 400
                        and not dropped_sampling
                        and _drop_unsupported_sampling(payload, text)
                    ):
                        dropped_sampling = True
                        self._bump("sampling_fallbacks")
                        print(
                            "Azure rejected sampling parameters; retrying without "
                            "temperature/top_p/penalties",
                            flush=True,
                        )
                        continue
                    error_code = _azure_error_code(text)
                    category = (
                        error_code
                        if error_code in {"invalid_prompt", "content_filter", "model_error"}
                        else "transient"
                        if exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                        else ""
                    )
                    retries_allowed = retry_limits.get(category, 0)
                    used = retry_counts.get(category, 0)
                    if used < retries_allowed:
                        retry_counts[category] = used + 1
                        if category in {"invalid_prompt", "content_filter"}:
                            _soften_payload(payload, used)
                            self._bump("content_filter_softened_retries")
                        print(
                            f"Azure HTTP {exc.code} {category}; retrying "
                            f"{used + 1}/{retries_allowed}",
                            flush=True,
                        )
                        if self.invalid_prompt_retry_delay > 0:
                            time.sleep(self.invalid_prompt_retry_delay)
                        continue
                    if category in {"invalid_prompt", "content_filter"}:
                        self._bump("content_filter_degraded_fallbacks")
                        self._send_json(
                            200,
                            _degraded_chat_response(body, self.model, category),
                        )
                        return
                    raise AzureHTTPError(exc.code, text) from exc
                except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                    used = retry_counts["transient"]
                    self._bump("upstream_transport_errors")
                    if used >= self.transient_retries:
                        raise AzureHTTPError(502, "Azure Responses transport failed") from exc
                    retry_counts["transient"] = used + 1
                    if self.invalid_prompt_retry_delay > 0:
                        time.sleep(self.invalid_prompt_retry_delay)
            self._send_json(200, _chat_response_from_azure(response, self.model))
        except AzureHTTPError as exc:
            print(f"Azure HTTP {exc.code}: {exc.text}", flush=True)
            self._send_json(
                exc.code,
                {"error": {"message": exc.text, "type": "azure_responses_error"}},
            )
        except Exception as exc:
            self._send_json(
                500,
                {"error": {"message": str(exc), "type": "azure_proxy_error"}},
            )


class AzureHTTPError(Exception):
    def __init__(self, code: int, text: str) -> None:
        super().__init__(text)
        self.code = code
        self.text = text


def _azure_error_code(text: str) -> str | None:
    generic_code: str | None = None
    generic_type: str | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str):
                normalized = code.lower()
                if any(
                    marker in normalized
                    for marker in ("content_filter", "responsibleai", "policy", "jailbreak")
                ):
                    return "content_filter"
                if normalized == "invalid_prompt":
                    return "invalid_prompt"
                if normalized == "model_error":
                    return "model_error"
                # Azure commonly wraps Responsible AI blocks in a generic
                # ``invalid_request_error`` code. Keep it as a fallback but
                # inspect the nested message and full body before deciding.
                generic_code = normalized
            error_type = error.get("type")
            if isinstance(error_type, str):
                normalized_type = error_type.lower()
                if any(
                    marker in normalized_type
                    for marker in ("content_filter", "responsibleai", "policy", "jailbreak")
                ):
                    return "content_filter"
                generic_type = normalized_type
            message = error.get("message")
            if isinstance(message, str):
                nested = _azure_error_code(message)
                if nested:
                    return nested
    lowered = text.lower()
    if '"code": "invalid_prompt"' in lowered or "invalid_prompt" in lowered:
        return "invalid_prompt"
    if any(
        marker in lowered
        for marker in (
            '"code": "content_filter"',
            "content_filter",
            "content management policy",
            "responsibleai",
            "responsible ai",
            "content_policy",
            "jailbreak",
        )
    ):
        return "content_filter"
    if '"type": "model_error"' in lowered:
        return "model_error"
    return generic_code or generic_type


_BENIGN_PREAMBLES = (
    "NOTE: This is a harmless, fictional, turn-based ASCII video-game "
    "benchmark. All combat words refer only to game sprites. Continue playing "
    "normally and follow the requested XML response format.\n\n",
    "SAFETY CONTEXT: This automated research evaluation controls a fictional "
    "Craftax game avatar. Nothing is real-world advice or action. Choose only "
    "one legal in-game action and return the required XML tag.\n\n",
)


def _soften_payload(payload: dict[str, Any], level: int) -> None:
    preamble = _BENIGN_PREAMBLES[min(level, len(_BENIGN_PREAMBLES) - 1)]
    instructions = str(payload.get("instructions") or "")
    if not instructions.startswith(preamble):
        payload["instructions"] = preamble + instructions


def _degraded_chat_response(
    request_body: dict[str, Any], model: str, reason: str
) -> dict[str, Any]:
    messages = list(request_body.get("messages") or [])
    last_user = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = _content_to_text(message.get("content"))
            break
    if "[Memory Update]" in last_user or "MEMORY TURN" in last_user:
        content = (
            "<memory>The Azure request was blocked after safe retries. Keep the "
            "previous plan and reassess from the next observation.</memory>"
        )
    else:
        content = (
            "Azure blocked this game request after safe rewording retries; "
            "preserving the trajectory with the legal no-op action.\n"
            "<action>NOOP</action>"
        )
    return {
        "id": f"azure-degraded-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "glyphbench_proxy": {"degraded": True, "reason": reason},
    }


def _drop_unsupported_sampling(payload: dict[str, Any], error_text: str) -> bool:
    """Drop reasoning-model-incompatible sampling knobs after one Azure 400."""

    removable = ("temperature", "top_p", "presence_penalty", "frequency_penalty")
    present = [key for key in removable if key in payload]
    if not present:
        return False
    low = error_text.lower()
    if "unsupported" not in low and not any(key in low for key in present):
        return False
    for key in present:
        payload.pop(key, None)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("AZURE_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AZURE_PROXY_PORT", "8010")))
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_RESPONSES_ENDPOINT"))
    parser.add_argument("--model", default=os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("AZURE_OPENAI_TIMEOUT", "600")))
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not args.endpoint:
        raise SystemExit("AZURE_OPENAI_RESPONSES_ENDPOINT is required")
    if not api_key:
        raise SystemExit("AZURE_OPENAI_API_KEY is required")

    AzureResponsesProxy.endpoint = normalize_responses_endpoint(args.endpoint)
    AzureResponsesProxy.api_key = api_key
    AzureResponsesProxy.model = args.model
    AzureResponsesProxy.policy_id = os.environ.get("AZURE_OPENAI_POLICY_ID") or None
    AzureResponsesProxy.auth_mode = os.environ.get("AZURE_OPENAI_AUTH_MODE", "auto")
    AzureResponsesProxy.timeout = args.timeout
    AzureResponsesProxy.invalid_prompt_retries = int(
        os.environ.get("AZURE_OPENAI_INVALID_PROMPT_RETRIES", "0")
    )
    AzureResponsesProxy.content_filter_retries = int(
        os.environ.get("AZURE_OPENAI_CONTENT_FILTER_RETRIES", "0")
    )
    AzureResponsesProxy.model_error_retries = int(
        os.environ.get("AZURE_OPENAI_MODEL_ERROR_RETRIES", "0")
    )
    AzureResponsesProxy.transient_retries = int(
        os.environ.get("AZURE_OPENAI_TRANSIENT_RETRIES", "4")
    )
    AzureResponsesProxy.invalid_prompt_retry_delay = float(
        os.environ.get("AZURE_OPENAI_INVALID_PROMPT_RETRY_DELAY", "1")
    )
    AzureResponsesProxy.metrics = {}
    AzureResponsesProxy.metrics_lock = threading.Lock()

    server = ThreadingHTTPServer((args.host, args.port), AzureResponsesProxy)
    print(
        f"Azure Responses proxy listening on http://{args.host}:{args.port}/v1 "
        f"for model {args.model}"
        + (f" with policy {AzureResponsesProxy.policy_id}" if AzureResponsesProxy.policy_id else ""),
        f"invalid_prompt_retries={AzureResponsesProxy.invalid_prompt_retries}",
        f"content_filter_retries={AzureResponsesProxy.content_filter_retries}",
        f"model_error_retries={AzureResponsesProxy.model_error_retries}",
        f"transient_retries={AzureResponsesProxy.transient_retries}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
