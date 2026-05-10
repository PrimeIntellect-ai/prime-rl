"""OpenAI-compatible proxy for prime-rl's Dynamo worker."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer


def _content_headers(headers: httpx.Headers) -> dict[str, str]:
    content_type = headers.get("content-type")
    if content_type is None:
        return {}
    return {"content-type": content_type}


class DynamoProxy:
    def __init__(self, upstream_url: str, worker_url: str, model: str, trust_remote_code: bool) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.worker_url = worker_url.rstrip("/")
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.client = httpx.AsyncClient(timeout=None)

    async def close(self) -> None:
        await self.client.aclose()

    def prompt_token_ids(self, body: dict[str, Any]) -> list[int]:
        messages = body["messages"]
        chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tools=body.get("tools"),
            documents=body.get("documents"),
            chat_template=body.get("chat_template"),
            add_generation_prompt=body.get("add_generation_prompt", True),
            continue_final_message=body.get("continue_final_message", False),
            tokenize=True,
            **chat_template_kwargs,
        )
        if isinstance(tokenized, Mapping):
            tokenized = tokenized["input_ids"]
        if tokenized and isinstance(tokenized[0], list):
            tokenized = tokenized[0]
        return list(tokenized)

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        if body.get("stream"):
            return JSONResponse(
                {"message": "Dynamo backend streaming chat completions are not supported by prime-rl."},
                status_code=400,
            )

        extra_body = body.pop("extra_body", None)
        if isinstance(extra_body, dict):
            body.update(extra_body)
        body.pop("return_token_ids", None)
        body["prompt_token_ids"] = self.prompt_token_ids(body)

        dp_rank = request.headers.get("x-data-parallel-rank")
        if dp_rank is not None:
            body.setdefault("routing", {})["dp_rank"] = int(dp_rank)

        response = await self.client.post(f"{self.worker_url}/engine/chat_completions", json=body)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=_content_headers(response.headers),
        )

    async def models(self) -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "prime-rl",
                    }
                ],
            }
        )

    async def forward(self, request: Request, path: str) -> Response:
        body = await request.body()
        response = await self.client.request(
            request.method,
            f"{self.upstream_url}/{path}",
            content=body,
            headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
            params=request.query_params,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=_content_headers(response.headers),
        )


def build_app(proxy: DynamoProxy) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await proxy.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def _models() -> JSONResponse:
        return await proxy.models()

    @app.post("/v1/chat/completions")
    async def _chat_completions(request: Request) -> Response:
        return await proxy.chat_completions(request)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def _forward(request: Request, path: str) -> Response:
        return await proxy.forward(request, path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="prime-rl Dynamo OpenAI proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    app = build_app(
        DynamoProxy(
            upstream_url=args.upstream_url,
            worker_url=args.worker_url,
            model=args.model,
            trust_remote_code=args.trust_remote_code,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
