import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from itertools import count
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


@dataclass(frozen=True)
class PDProxyConfig:
    host: str
    port: int
    prefill_urls: tuple[str, ...]
    decode_urls: tuple[str, ...]
    prefill_kv_addrs: tuple[str, ...]
    decode_kv_addrs: tuple[str, ...]
    timeout: float
    max_generate_attempts: int = 2
    health_check_interval: float = 5.0
    health_check_timeout: float = 5.0
    health_check_failure_threshold: int = 2


def build_request_id(prefill_kv_addr: str, decode_kv_addr: str) -> str:
    return f"___prefill_addr_{prefill_kv_addr}___decode_addr_{decode_kv_addr}_{uuid4().hex}"


def build_prefill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prefill_payload = payload.copy()
    prefill_payload["max_tokens"] = 1
    if "max_completion_tokens" in prefill_payload:
        prefill_payload["max_completion_tokens"] = 1
    return prefill_payload


def create_app(
    config: PDProxyConfig,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if len(config.prefill_urls) == 0:
        raise ValueError("prefill_urls must not be empty.")
    if len(config.decode_urls) == 0:
        raise ValueError("decode_urls must not be empty.")
    if len(config.prefill_urls) != len(config.prefill_kv_addrs):
        raise ValueError("prefill_urls and prefill_kv_addrs must have the same length.")
    if len(config.decode_urls) != len(config.decode_kv_addrs):
        raise ValueError("decode_urls and decode_kv_addrs must have the same length.")
    if config.max_generate_attempts < 1:
        raise ValueError("max_generate_attempts must be >= 1.")
    if config.health_check_failure_threshold < 1:
        raise ValueError("health_check_failure_threshold must be >= 1.")
    if config.health_check_interval <= 0:
        raise ValueError("health_check_interval must be > 0.")
    if config.health_check_timeout <= 0:
        raise ValueError("health_check_timeout must be > 0.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout), transport=transport)
        health_task = asyncio.create_task(_health_check_loop())
        yield
        health_task.cancel()
        with suppress(asyncio.CancelledError):
            await health_task
        await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    prefill_route_counter = count()
    decode_route_counter = count()
    backend_urls = (*config.prefill_urls, *config.decode_urls)
    prefill_kv_by_url = dict(zip(config.prefill_urls, config.prefill_kv_addrs, strict=True))
    decode_kv_by_url = dict(zip(config.decode_urls, config.decode_kv_addrs, strict=True))
    backend_failure_counts = {url: 0 for url in backend_urls}
    dead_backends: set[str] = set()

    async def _read_json_object(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return payload

    def _passthrough(response: httpx.Response) -> Response:
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    async def _request(request_call) -> httpx.Response:
        response = await request_call
        response.raise_for_status()
        return response

    async def _request_or_502(request_call, context: str) -> httpx.Response:
        try:
            response = await _request(request_call)
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502, detail=f"{context} failed with status {e.response.status_code}."
            ) from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"{context} request failed: {e}") from e
        return response

    def _build_headers(request: Request, request_id: str, decode_kv_addr: str) -> dict[str, str]:
        headers = {
            "X-Request-Id": request_id,
            "X-KV-Target": decode_kv_addr,
        }
        auth = request.headers.get("authorization")
        if auth is not None:
            headers["Authorization"] = auth
        return headers

    def _mark_backend_success(url: str) -> None:
        backend_failure_counts[url] = 0
        dead_backends.discard(url)

    def _mark_backend_failure(url: str) -> None:
        failure_count = backend_failure_counts[url] + 1
        backend_failure_counts[url] = failure_count
        if failure_count >= config.health_check_failure_threshold:
            dead_backends.add(url)

    def _rotated(urls: tuple[str, ...], idx: int) -> tuple[str, ...]:
        if len(urls) == 0:
            return ()
        offset = idx % len(urls)
        return urls[offset:] + urls[:offset]

    def _healthy_backends(urls: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(url for url in urls if url not in dead_backends)

    def _select_backend_pair(tried_pairs: set[tuple[str, str]]) -> tuple[str, str] | None:
        prefill_candidates = _healthy_backends(config.prefill_urls)
        decode_candidates = _healthy_backends(config.decode_urls)
        if len(prefill_candidates) == 0 or len(decode_candidates) == 0:
            return None

        prefill_order = _rotated(prefill_candidates, next(prefill_route_counter))
        decode_order = _rotated(decode_candidates, next(decode_route_counter))
        for prefill_url in prefill_order:
            for decode_url in decode_order:
                pair = (prefill_url, decode_url)
                if pair not in tried_pairs:
                    return pair
        return None

    async def _check_backend_health(url: str) -> None:
        try:
            response = await app.state.client.get(f"{url}/health", timeout=config.health_check_timeout)
        except httpx.HTTPError:
            _mark_backend_failure(url)
            return

        if response.status_code >= 400:
            _mark_backend_failure(url)
            return
        _mark_backend_success(url)

    async def _health_check_loop() -> None:
        while True:
            await asyncio.sleep(config.health_check_interval)
            await asyncio.gather(*(_check_backend_health(url) for url in backend_urls))

    def _is_retryable_status(status_code: int) -> bool:
        return status_code >= 500

    def _request_targets_base(request_url: httpx.URL, base_url: str) -> bool:
        base = httpx.URL(base_url)
        return request_url.host == base.host and request_url.port == base.port

    async def _generate_once(
        path: str,
        payload: dict[str, Any],
        prefill_payload: dict[str, Any],
        prefill_url: str,
        decode_url: str,
        request: Request,
    ) -> Response:
        prefill_kv_addr = prefill_kv_by_url[prefill_url]
        decode_kv_addr = decode_kv_by_url[decode_url]
        request_id = build_request_id(prefill_kv_addr, decode_kv_addr)
        headers = _build_headers(request, request_id, decode_kv_addr)
        await _request(app.state.client.post(f"{prefill_url}{path}", json=prefill_payload, headers=headers))
        decode_response = await _request(app.state.client.post(f"{decode_url}{path}", json=payload, headers=headers))
        return _passthrough(decode_response)

    def _select_backends_for_request(tried_pairs: set[tuple[str, str]]) -> tuple[str, str]:
        pair = _select_backend_pair(tried_pairs)
        if pair is None:
            raise HTTPException(status_code=503, detail="No healthy PD backends available.")
        return pair

    def _failed_backend_for_request(
        request_url: httpx.URL | None,
        prefill_url: str,
        decode_url: str,
    ) -> str:
        if request_url is not None and _request_targets_base(request_url, prefill_url):
            return prefill_url
        return decode_url

    async def _post_json(
        base_url: str,
        path: str,
        payload: dict[str, Any],
        context: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await _request_or_502(
            app.state.client.post(f"{base_url}{path}", json=payload, headers=headers),
            context=context,
        )

    async def _fanout_admin(path: str, payload: dict[str, Any]) -> None:
        await asyncio.gather(*(_post_json(url, path, payload, f"{path} {url}") for url in backend_urls))

    async def _fanout_admin_per_url(path: str, payloads_by_url: list[tuple[str, dict[str, Any]]]) -> None:
        await asyncio.gather(*(_post_json(url, path, payload, f"{path} {url}") for url, payload in payloads_by_url))

    async def _fanout_admin_from_request(path: str, request: Request) -> dict[str, str]:
        payload = await _read_json_object(request)
        await _fanout_admin(path, payload)
        return {"status": "ok"}

    async def _handle_generate(request: Request, upstream_path: str | None = None) -> Response:
        payload = await _read_json_object(request)
        prefill_payload = build_prefill_payload(payload)
        tried_pairs: set[tuple[str, str]] = set()
        last_error_detail = "No backend attempts were made."
        path = upstream_path or request.url.path

        for _ in range(config.max_generate_attempts):
            prefill_url, decode_url = _select_backends_for_request(tried_pairs)
            tried_pairs.add((prefill_url, decode_url))

            try:
                response = await _generate_once(
                    path,
                    payload,
                    prefill_payload,
                    prefill_url,
                    decode_url,
                    request,
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if not _is_retryable_status(status_code):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Backend returned non-retryable status {status_code}.",
                    ) from e

                failed_url = _failed_backend_for_request(e.request.url, prefill_url, decode_url)
                _mark_backend_failure(failed_url)
                last_error_detail = f"{failed_url} returned {status_code}"
                continue
            except httpx.HTTPError as e:
                request_url = e.request.url if isinstance(e, httpx.RequestError) and e.request is not None else None
                failed_url = _failed_backend_for_request(request_url, prefill_url, decode_url)
                _mark_backend_failure(failed_url)
                last_error_detail = f"{failed_url} request failed: {e}"
                continue

            _mark_backend_success(prefill_url)
            _mark_backend_success(decode_url)
            return response

        raise HTTPException(status_code=502, detail=f"All PD backend attempts failed. Last error: {last_error_detail}")

    @app.get("/health")
    async def health():
        try:
            responses = await asyncio.gather(*(app.state.client.get(f"{url}/health") for url in backend_urls))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Health check failed: {e}") from e

        if any(response.status_code >= 400 for response in responses):
            raise HTTPException(status_code=503, detail="One or more PD backends are unhealthy.")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        response = await _request_or_502(
            app.state.client.get(f"{config.decode_urls[0]}/v1/models"),
            context="/v1/models decode",
        )
        return _passthrough(response)

    @app.post("/update_weights")
    async def update_weights(request: Request):
        return await _fanout_admin_from_request("/update_weights", request)

    @app.post("/reload_weights")
    async def reload_weights():
        await _fanout_admin("/reload_weights", {})
        return {"status": "ok"}

    @app.post("/load_lora_adapter")
    async def load_lora_adapter(request: Request):
        return await _fanout_admin_from_request("/load_lora_adapter", request)

    @app.post("/init_broadcaster")
    async def init_broadcaster(request: Request):
        payload = await _read_json_object(request)
        num_inference_server = len(backend_urls)
        await _fanout_admin_per_url(
            "/init_broadcaster",
            [
                (
                    url,
                    {
                        **payload,
                        "server_rank": server_rank,
                        "num_inference_server": num_inference_server,
                    },
                )
                for server_rank, url in enumerate(backend_urls)
            ],
        )
        return {"status": "ok"}

    @app.post("/v1/chat/completions/tokens")
    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def generate_route(request: Request):
        return await _handle_generate(request)

    @app.post("/chat/completions/tokens")
    async def chat_completions_tokens_root(request: Request):
        # verifiers' token client strips trailing /v1 from base_url and calls this root path.
        return await _handle_generate(request, upstream_path="/v1/chat/completions/tokens")

    return app


def parse_args() -> PDProxyConfig:
    def _parse_csv(raw: str | None) -> tuple[str, ...]:
        if raw is None:
            return ()
        return tuple(item.strip() for item in raw.split(",") if item.strip())

    def _require_non_empty(values: tuple[str, ...], arg_name: str) -> None:
        if len(values) == 0:
            parser.error(f"Provide --{arg_name}.")

    def _require_matching_counts(left: tuple[str, ...], right: tuple[str, ...], message: str) -> None:
        if len(left) != len(right):
            parser.error(message)

    parser = argparse.ArgumentParser(description="PD disaggregation proxy for prime-rl.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-urls", type=str, help="Comma-separated prefill backend URLs.")
    parser.add_argument("--decode-urls", type=str, help="Comma-separated decode backend URLs.")
    parser.add_argument("--prefill-kv-addrs", type=str, help="Comma-separated prefill KV addresses.")
    parser.add_argument("--decode-kv-addrs", type=str, help="Comma-separated decode KV addresses.")
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--max-generate-attempts", type=int, default=2)
    parser.add_argument("--health-check-interval", type=float, default=5.0)
    parser.add_argument("--health-check-timeout", type=float, default=5.0)
    parser.add_argument("--health-check-failure-threshold", type=int, default=2)
    args = parser.parse_args()

    prefill_urls = _parse_csv(args.prefill_urls)
    decode_urls = _parse_csv(args.decode_urls)
    prefill_kv_addrs = _parse_csv(args.prefill_kv_addrs)
    decode_kv_addrs = _parse_csv(args.decode_kv_addrs)

    _require_non_empty(prefill_urls, "prefill-urls")
    _require_non_empty(decode_urls, "decode-urls")
    _require_non_empty(prefill_kv_addrs, "prefill-kv-addrs")
    _require_non_empty(decode_kv_addrs, "decode-kv-addrs")
    _require_matching_counts(prefill_urls, prefill_kv_addrs, "prefill URL and KV address counts must match.")
    _require_matching_counts(decode_urls, decode_kv_addrs, "decode URL and KV address counts must match.")

    return PDProxyConfig(
        host=args.host,
        port=args.port,
        prefill_urls=tuple(url.rstrip("/") for url in prefill_urls),
        decode_urls=tuple(url.rstrip("/") for url in decode_urls),
        prefill_kv_addrs=prefill_kv_addrs,
        decode_kv_addrs=decode_kv_addrs,
        timeout=args.timeout,
        max_generate_attempts=args.max_generate_attempts,
        health_check_interval=args.health_check_interval,
        health_check_timeout=args.health_check_timeout,
        health_check_failure_threshold=args.health_check_failure_threshold,
    )


def main():
    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
