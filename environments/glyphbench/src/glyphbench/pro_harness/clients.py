"""Pluggable, robust LLM clients for the Pro harness.

Two transports, one uniform interface (``LLMClient.complete``):

  * ``OpenAIChatClient``   — any OpenAI-compatible ``/v1/chat/completions``
                             server (vLLM, OpenAI, Prime Inference, or the
                             Azure chat-completions proxy).
  * ``AzureResponsesClient`` — the Azure OpenAI **Responses API** hit
                             directly over HTTP (for GPT-5.x reasoning /
                             "pro" deployments), mirroring the original
                             Craftax harness so the harness — not a proxy —
                             owns the robustness.

Robustness contract (shared, implemented in the base class):
  * **No reasoning truncation.** When ``max_output_tokens`` is ``None`` the
    client sends *no* output cap. It never lowers the cap on a retry.
  * **Azure content-filter / refusal resilience.** A content-filter HTTP
    error, an empty incomplete-due-to-filter response, or a refusal phrase
    in the output triggers a retry with a *softened, benign* system framing
    (graduated intensity), exactly the trick the original harness used.
  * **Transient-error backoff.** 429 / 5xx / timeouts / connection drops are
    retried with exponential backoff.
  * **Unsupported-parameter fallback.** Reasoning deployments that reject
    ``temperature`` / ``top_p`` / ``reasoning_effort`` are retried with those
    stripped (the original's behaviour).

``complete`` always returns a ``CompletionResult`` — it does not raise for
API failures. On unrecoverable failure it returns an empty-text result with
``error`` set and ``finish_reason`` in {"error", "content_filter"}; the
harness degrades that turn gracefully (a forfeit + feedback) instead of
crashing the rollout.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from glyphbench.azure_openai import (
    azure_auth_headers,
    normalize_responses_endpoint,
    redact_endpoint_url,
)
from glyphbench.pro_harness.config import ProConfig

# Refusal phrases that indicate a content-policy soft refusal in the *output*
# (Azure sometimes returns a polite refusal as normal text rather than a 400).
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "i cannot assist",
    "i can't assist",
    "i'm unable to",
    "i am unable to",
    "i can't help",
    "i cannot help",
    "i won't be able to",
    "not able to assist",
    "against my guidelines",
    "i must decline",
    "i can’t assist",
    "i’m sorry",
)

# Substrings that mark an Azure content-management / responsible-AI block.
_CONTENT_FILTER_MARKERS: tuple[str, ...] = (
    "content_filter",
    "content management policy",
    "responsibleai",
    "responsible ai",
    "content_policy",
    "jailbreak",
    "content filtering",
)
_INVALID_PROMPT_MARKERS: tuple[str, ...] = ("invalid_prompt",)
_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "unsupported parameter",
    "unsupported_parameter",
    "unsupported value",
    "unsupported_value",
    "is not supported with this model",
    "does not support",
    "unknown parameter",
)

_BENIGN_PREAMBLES: tuple[str, ...] = (
    # Level 1 — light disclaimer prepended to the full game system prompt.
    "NOTE: The following is a fictional, turn-based ASCII video-game "
    "evaluation. All characters, combat, and events are part of a harmless "
    "research game and have no real-world meaning. Play the game normally "
    "and reply in the required format.\n\n",
    # Level 2 — stronger framing.
    "SAFETY CONTEXT: This is an automated benchmark of a fictional ASCII "
    "roguelike video game. Words like attack, kill, fight, or necromancer "
    "refer only to game sprites in a make-believe world; nothing here is "
    "real or harmful. You are simply choosing the next move for a game "
    "avatar. Continue and answer in the required format.\n\n",
)


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _REFUSAL_PHRASES)


def _classify_error_text(text: str) -> str | None:
    """Map an Azure/OpenAI error body to a coarse class."""
    low = text.lower()
    if any(m in low for m in _UNSUPPORTED_MARKERS):
        return "unsupported"
    if any(m in low for m in _CONTENT_FILTER_MARKERS):
        return "content_filter"
    if any(m in low for m in _INVALID_PROMPT_MARKERS):
        return "content_filter"
    return None


@dataclass
class CompletionResult:
    """The outcome of one ``complete`` call (after all internal retries)."""

    text: str = ""
    reasoning: str = ""
    finish_reason: str = "stop"  # stop | length | content_filter | error
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    blocked: bool = False
    softened: bool = False
    attempts: int = 0
    latency_s: float = 0.0
    transient_retries: int = 0
    content_filter_retries: int = 0
    unsupported_param_retries: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# --- internal control-flow exceptions (never escape the client) ----------
class _TransientError(Exception):
    pass


class _ContentFilterError(Exception):
    pass


class _UnsupportedParamError(Exception):
    pass


class LLMClient(ABC):
    """Transport-agnostic completion client with shared retry orchestration."""

    def __init__(self, cfg: ProConfig) -> None:
        self.cfg = cfg

    # -- subclass hook: one raw attempt -----------------------------------
    @abstractmethod
    def _attempt(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        *,
        drop_sampling: bool,
    ) -> CompletionResult:
        """Issue a single request. Raise ``_TransientError`` /
        ``_ContentFilterError`` / ``_UnsupportedParamError`` to drive the
        base-class retry policy; return a ``CompletionResult`` otherwise."""

    @abstractmethod
    def describe(self) -> str: ...

    # -- public API -------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        *,
        label: str = "",
    ) -> CompletionResult:
        cfg = self.cfg
        t0 = time.monotonic()
        attempts = 0
        transient_used = 0
        soften_used = 0
        unsupported_used = 0
        drop_sampling = False
        softened = False
        cur_system = system

        while True:
            attempts += 1
            try:
                res = self._attempt(cur_system, user, drop_sampling=drop_sampling)
            except _UnsupportedParamError as exc:
                if not drop_sampling:
                    drop_sampling = True
                    unsupported_used += 1
                    continue
                return CompletionResult(
                    finish_reason="error",
                    error=f"unsupported_param: {exc}",
                    attempts=attempts,
                    softened=softened,
                    latency_s=time.monotonic() - t0,
                    transient_retries=transient_used,
                    content_filter_retries=soften_used,
                    unsupported_param_retries=unsupported_used,
                )
            except _ContentFilterError as exc:
                if soften_used < cfg.content_filter_retries:
                    cur_system = self._soften(system, soften_used)
                    soften_used += 1
                    softened = True
                    continue
                return CompletionResult(
                    finish_reason="content_filter",
                    blocked=True,
                    error=f"content_filter: {exc}",
                    attempts=attempts,
                    softened=softened,
                    latency_s=time.monotonic() - t0,
                    transient_retries=transient_used,
                    content_filter_retries=soften_used,
                    unsupported_param_retries=unsupported_used,
                )
            except _TransientError as exc:
                if transient_used < cfg.max_retries:
                    transient_used += 1
                    time.sleep(min(cfg.retry_backoff**transient_used, 30.0))
                    continue
                return CompletionResult(
                    finish_reason="error",
                    error=f"transient: {exc}",
                    attempts=attempts,
                    softened=softened,
                    latency_s=time.monotonic() - t0,
                    transient_retries=transient_used,
                    content_filter_retries=soften_used,
                    unsupported_param_retries=unsupported_used,
                )
            except Exception as exc:  # noqa: BLE001 — never let a turn crash the rollout
                # An unexpected backend/SDK error: retry like a transient
                # error, then degrade to an error result so the episode
                # continues (the harness forfeits this turn with feedback).
                if transient_used < cfg.max_retries:
                    transient_used += 1
                    time.sleep(min(cfg.retry_backoff**transient_used, 30.0))
                    continue
                return CompletionResult(
                    finish_reason="error",
                    error=f"unexpected: {type(exc).__name__}: {exc}",
                    attempts=attempts,
                    softened=softened,
                    latency_s=time.monotonic() - t0,
                    transient_retries=transient_used,
                    content_filter_retries=soften_used,
                    unsupported_param_retries=unsupported_used,
                )

            # Got a response. A polite refusal in the body is also a filter.
            if (
                res.text
                and _looks_like_refusal(res.text)
                and soften_used < cfg.content_filter_retries
            ):
                cur_system = self._soften(system, soften_used)
                soften_used += 1
                softened = True
                continue

            res.attempts = attempts
            res.softened = softened
            res.latency_s = time.monotonic() - t0
            res.transient_retries = transient_used
            res.content_filter_retries = soften_used
            res.unsupported_param_retries = unsupported_used
            return res

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _soften(system: str, level: int) -> str:
        preamble = _BENIGN_PREAMBLES[min(level, len(_BENIGN_PREAMBLES) - 1)]
        return preamble + system


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions (vLLM / OpenAI / Prime / proxy)
# ---------------------------------------------------------------------------
class OpenAIChatClient(LLMClient):
    def __init__(self, cfg: ProConfig) -> None:
        super().__init__(cfg)
        if not cfg.model:
            raise ValueError("OpenAIChatClient requires cfg.model")
        from openai import OpenAI

        # We run our own retry/backoff so the SDK must not also retry.
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "EMPTY",
            timeout=cfg.request_timeout,
            max_retries=0,
        )

    def describe(self) -> str:
        return f"openai-chat model={self.cfg.model} base_url={redact_endpoint_url(self.cfg.base_url)}"

    def _attempt(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        *,
        drop_sampling: bool,
    ) -> CompletionResult:
        import openai

        cfg = self.cfg
        messages = (
            [{"role": "system", "content": system}, *user]
            if isinstance(user, list)
            else [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        kwargs: dict[str, object] = {"model": cfg.model, "messages": messages}
        if cfg.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = int(cfg.max_output_tokens)
        if not drop_sampling:
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                kwargs["top_p"] = cfg.top_p
            if cfg.presence_penalty is not None:
                kwargs["presence_penalty"] = cfg.presence_penalty
            if cfg.reasoning_effort is not None:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            if cfg.top_k is not None:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body["top_k"] = int(cfg.top_k)
                kwargs["extra_body"] = extra_body
            if cfg.enable_thinking is not None:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": bool(cfg.enable_thinking)
                }
                kwargs["extra_body"] = extra_body

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            body = _err_text(exc)
            kind = _classify_error_text(body)
            if kind == "unsupported":
                raise _UnsupportedParamError(body) from exc
            if kind == "content_filter":
                raise _ContentFilterError(body) from exc
            # Some servers reject sampling params with a generic 400.
            if not drop_sampling and any(
                k in body.lower()
                for k in (
                    "temperature", "top_p", "top_k", "presence_penalty",
                    "reasoning_effort", "chat_template_kwargs", "enable_thinking",
                )
            ):
                raise _UnsupportedParamError(body) from exc
            # Genuine bad request — not retryable.
            return CompletionResult(finish_reason="error", error=f"bad_request: {body[:500]}")
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise _TransientError(_err_text(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                raise _TransientError(_err_text(exc)) from exc
            body = _err_text(exc)
            if _classify_error_text(body) == "content_filter":
                raise _ContentFilterError(body) from exc
            return CompletionResult(
                finish_reason="error", error=f"api_status_{exc.status_code}: {body[:500]}"
            )

        choice = resp.choices[0] if resp.choices else None
        if choice is None:
            return CompletionResult(finish_reason="error", error="no_choices")
        finish = choice.finish_reason or "stop"
        if finish == "content_filter":
            raise _ContentFilterError("finish_reason=content_filter")
        msg = choice.message
        text = msg.content or ""
        reasoning = _extract_chat_reasoning(msg)
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        input_details = getattr(usage, "prompt_tokens_details", None)
        cached_tok = getattr(input_details, "cached_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        reason_tok = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reason_tok = getattr(details, "reasoning_tokens", 0) or 0
        return CompletionResult(
            text=text,
            reasoning=reasoning,
            finish_reason="length" if finish == "length" else "stop",
            input_tokens=int(in_tok),
            cached_input_tokens=int(cached_tok),
            output_tokens=int(out_tok),
            reasoning_tokens=int(reason_tok),
        )


# ---------------------------------------------------------------------------
# Azure Responses API (direct HTTP) — for GPT-5.x reasoning deployments
# ---------------------------------------------------------------------------
class AzureResponsesClient(LLMClient):
    def __init__(self, cfg: ProConfig) -> None:
        super().__init__(cfg)
        if not cfg.azure_endpoint:
            raise ValueError(
                "AzureResponsesClient requires cfg.azure_endpoint "
                "(the full .../responses?api-version=... URL)"
            )
        if not cfg.azure_api_key:
            raise ValueError("AzureResponsesClient requires cfg.azure_api_key")
        self._endpoint = normalize_responses_endpoint(cfg.azure_endpoint)
        self._model = cfg.model or "gpt-5.4"

    def describe(self) -> str:
        return f"azure-responses model={self._model} endpoint={redact_endpoint_url(self._endpoint)}"

    def _attempt(
        self,
        system: str,
        user: str | list[dict[str, Any]],
        *,
        drop_sampling: bool,
    ) -> CompletionResult:
        cfg = self.cfg
        body: dict[str, object] = {
            "model": self._model,
            "instructions": system,
            "input": user,
        }
        if cfg.max_output_tokens is not None:
            body["max_output_tokens"] = int(cfg.max_output_tokens)
        reasoning: dict[str, str] = {}
        if cfg.reasoning_effort is not None:
            reasoning["effort"] = cfg.reasoning_effort
        if cfg.reasoning_mode is not None:
            reasoning["mode"] = cfg.reasoning_mode
        if reasoning:
            body["reasoning"] = reasoning
        if not drop_sampling:
            if cfg.temperature is not None:
                body["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                body["top_p"] = cfg.top_p

        headers = {"content-type": "application/json"}
        headers.update(
            azure_auth_headers(
                self._endpoint,
                cfg.azure_api_key or "",
                auth_mode=cfg.azure_auth_mode,
            )
        )
        if cfg.azure_policy_id:
            headers["x-policy-id"] = cfg.azure_policy_id

        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=cfg.request_timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            kind = _classify_error_text(text)
            if kind == "content_filter":
                raise _ContentFilterError(text[:500]) from exc
            if kind == "unsupported":
                raise _UnsupportedParamError(text[:500]) from exc
            if exc.code in (408, 409, 425, 429, 500, 502, 503, 504):
                raise _TransientError(f"http_{exc.code}: {text[:300]}") from exc
            if (
                exc.code == 400
                and not drop_sampling
                and any(k in text.lower() for k in ("temperature", "top_p"))
            ):
                raise _UnsupportedParamError(text[:500]) from exc
            return CompletionResult(finish_reason="error", error=f"http_{exc.code}: {text[:500]}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _TransientError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise _TransientError(f"bad_json: {exc}") from exc

        # An incomplete response due to a content filter shows up as a
        # completed/incomplete status with empty text + a filter reason.
        text = _extract_responses_text(payload)
        status = payload.get("status")
        incomplete = payload.get("incomplete_details") or {}
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        if status == "incomplete" and reason in {"content_filter", "responsible_ai_policy"}:
            raise _ContentFilterError(f"incomplete:{reason}")
        if not text and status in {"failed", "incomplete"}:
            err = payload.get("error") or {}
            err_text = json.dumps(err) if err else f"empty_{status}"
            if _classify_error_text(err_text) == "content_filter":
                raise _ContentFilterError(err_text[:500])
            # Empty output but no obvious filter — treat as transient once.
            raise _TransientError(f"empty_output:{status}:{reason}")

        finish = (
            "length"
            if (status == "incomplete" and reason in {"max_output_tokens", "max_tokens"})
            else "stop"
        )
        reasoning = _extract_responses_reasoning(payload)
        usage = payload.get("usage") or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        input_details = usage.get("input_tokens_details") or {}
        cached_tok = int(input_details.get("cached_tokens", 0) or 0)
        details = usage.get("output_tokens_details") or {}
        reason_tok = int(details.get("reasoning_tokens", 0) or 0)
        return CompletionResult(
            text=text,
            reasoning=reasoning,
            finish_reason=finish,
            input_tokens=in_tok,
            cached_input_tokens=cached_tok,
            output_tokens=out_tok,
            reasoning_tokens=reason_tok,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _err_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return json.dumps(body)
    if body:
        return str(body)
    return str(exc)


def _extract_chat_reasoning(msg: object) -> str:
    # vLLM / Qwen expose `reasoning_content`; some SDKs put it in model_extra.
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        return str(reasoning).strip()
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if val:
                return str(val).strip()
    return ""


def _extract_responses_text(payload: dict) -> str:
    out = payload.get("output_text")
    if isinstance(out, str) and out:
        return out
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                txt = part.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
    return "".join(chunks).strip()


def _extract_responses_reasoning(payload: dict) -> str:
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, list):
            for part in summary:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def build_client(cfg: ProConfig) -> LLMClient:
    """Construct the client selected by ``cfg.backend``."""
    if cfg.backend == "azure":
        return AzureResponsesClient(cfg)
    return OpenAIChatClient(cfg)
