"""ASGI middleware that HMAC-signs vLLM usage data in chat completion responses.

Signs ``{run_id}:{prompt_tokens}:{completion_tokens}`` and injects the
signature as ``usage_signature`` in the response JSON. The orchestrator
verifies signatures to detect tampering by untrusted verifiers environments.
"""

import json
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from prime_rl.utils.usage_signing import sign_usage

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class UsageSigningMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, secret: str) -> None:
        super().__init__(app)
        self._secret = secret

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path != CHAT_COMPLETIONS_PATH:
            return await call_next(request)

        run_id = request.headers.get("x-run-id")
        if not run_id:
            return JSONResponse(
                status_code=400,
                content={"error": "X-Run-Id header is required"},
            )

        response = await call_next(request)

        if response.status_code != 200:
            return response

        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        body_bytes = b"".join(chunks)

        try:
            data: dict = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(content=body_bytes, status_code=response.status_code, headers=dict(response.headers))

        usage = data.get("usage")
        if usage and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            data["usage_signature"] = sign_usage(
                self._secret,
                run_id,
                prompt_tokens,
                completion_tokens,
            )

        signed_body = json.dumps(data).encode()
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return Response(
            content=signed_body,
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
        )
