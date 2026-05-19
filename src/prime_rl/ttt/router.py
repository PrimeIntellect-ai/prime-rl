from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from prime_rl.utils.logger import get_logger, setup_logger

SESSION_ENDPOINTS = {
    "start_session",
    "prepare_turn",
    "complete_turn",
    "finish_session",
    "abort_session",
}


def _shard_index(session_id: str, num_shards: int) -> int:
    digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def _proxy_response(response: httpx.Response) -> Response:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return JSONResponse(content=response.json(), status_code=response.status_code)
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type or None,
    )


def create_app(learner_urls: list[str], request_timeout_s: float) -> FastAPI:
    app = FastAPI(title="Prime-RL TTT learner router")
    shards = [url.rstrip("/") for url in learner_urls]
    logger = get_logger()

    if not shards:
        raise ValueError("TTT router requires at least one learner URL.")

    def route_for(session_id: str) -> tuple[int, str]:
        idx = _shard_index(session_id, len(shards))
        return idx, shards[idx]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=request_timeout_s) as client:
            results = await asyncio.gather(
                *(client.get(f"{url}/health") for url in shards),
                return_exceptions=True,
            )
        shard_status: list[dict[str, Any]] = []
        unhealthy: list[str] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                unhealthy.append(f"{idx}:{result}")
                shard_status.append({"idx": idx, "url": shards[idx], "status": "error", "error": str(result)})
                continue
            try:
                body = result.json()
            except Exception:
                body = {"text": result.text}
            if result.status_code >= 400:
                unhealthy.append(f"{idx}:HTTP {result.status_code}")
            shard_status.append(
                {
                    "idx": idx,
                    "url": shards[idx],
                    "status_code": result.status_code,
                    "body": body,
                }
            )
        if unhealthy:
            raise HTTPException(status_code=503, detail={"unhealthy": unhealthy, "shards": shard_status})
        return {
            "status": "ok",
            "num_shards": len(shards),
            "shards": shard_status,
            "elapsed_s": time.perf_counter() - start,
        }

    @app.post("/update_base_weights")
    async def update_base_weights(request: Request) -> dict[str, Any]:
        start = time.perf_counter()
        payload = await request.json()
        async with httpx.AsyncClient(timeout=request_timeout_s) as client:
            results = await asyncio.gather(
                *(client.post(f"{url}/update_base_weights", json=payload) for url in shards),
                return_exceptions=True,
            )
        responses: list[dict[str, Any]] = []
        failures: list[str] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(f"{idx}:{result}")
                responses.append({"idx": idx, "url": shards[idx], "status": "error", "error": str(result)})
                continue
            try:
                body = result.json()
            except Exception:
                body = {"text": result.text}
            if result.status_code >= 400:
                failures.append(f"{idx}:HTTP {result.status_code}")
            responses.append(
                {
                    "idx": idx,
                    "url": shards[idx],
                    "status_code": result.status_code,
                    "body": body,
                }
            )
        if failures:
            raise HTTPException(status_code=503, detail={"failures": failures, "responses": responses})
        logger.info(
            f"TTT router broadcast update_base_weights step={payload.get('step')} "
            f"shards={len(shards)} elapsed={time.perf_counter() - start:.3f}s"
        )
        return {"status": "ok", "shards": responses}

    @app.post("/{endpoint}")
    async def route_session_endpoint(endpoint: str, request: Request) -> Response:
        if endpoint not in SESSION_ENDPOINTS:
            raise HTTPException(status_code=404, detail=f"Unknown TTT router endpoint: {endpoint}")
        payload = await request.json()
        session_id = payload.get("session_id")
        if not session_id:
            raise HTTPException(status_code=422, detail=f"/{endpoint} requires a session_id field.")
        idx, shard_url = route_for(str(session_id))
        async with httpx.AsyncClient(timeout=request_timeout_s) as client:
            response = await client.post(f"{shard_url}/{endpoint}", json=payload)
        logger.debug(f"TTT router {endpoint} session={session_id} shard={idx} status={response.status_code}")
        return _proxy_response(response)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--learner-url", action="append", required=True)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    get_logger().info(f"Starting TTT router on {args.host}:{args.port} for {len(args.learner_url)} learner shard(s)")
    uvicorn.run(
        create_app(args.learner_url, request_timeout_s=args.request_timeout_s),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
