#!/usr/bin/env python3
"""
Stub HTTP server for prime-rl admin endpoints.

NOTE: As of dynamo#8630 (bis/parity-tokenize-tcp), Dynamo's Rust frontend
implements these routes natively at /v1/rl/* when DYN_ENABLE_RL=true:
  POST /v1/rl/load_lora_adapter
  POST /v1/rl/unload_lora_adapter
  GET  /v1/rl/health

For K8s and any deployment with a real Dynamo frontend, point admin_base_url
at the Dynamo service (e.g. http://<frontend-svc>:8000/v1/rl). This stub is
kept as a local development fallback for running the orchestrator without a
live Dynamo instance.

Usage:
    python tools/dynamo/admin_stub.py
    python tools/dynamo/admin_stub.py --port 8001
"""

import argparse

from aiohttp import web


async def pause(request):
    print("[stub] POST /pause - OK")
    return web.Response(status=200, text="OK")


async def resume(request):
    print("[stub] POST /resume - OK")
    return web.Response(status=200, text="OK")


async def update_weights(request):
    body = await request.json()
    print(f"[stub] POST /update_weights weight_dir={body.get('weight_dir')} - OK (weights not reloaded)")
    return web.Response(status=200, text="OK")


async def health(request):
    return web.Response(status=200, text="OK")


app = web.Application()
app.router.add_post("/pause", pause)
app.router.add_post("/resume", resume)
app.router.add_post("/update_weights", update_weights)
app.router.add_get("/health", health)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo admin stub server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    args = parser.parse_args()
    print(f"[stub] Dynamo admin stub server starting on port {args.port}...")
    web.run_app(app, port=args.port)
