from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvloop
import vllm.envs as envs
from fastapi import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from prime_rl.inference.config import InferenceConfig

logger = init_logger("vllm.entrypoints.openai.api_server")


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we extend the engine args with our custom worker extension
@asynccontextmanager
async def custom_build_async_engine_client(
    args: Namespace,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"

    async with build_async_engine_client_from_engine_args(
        engine_args, args.disable_frontend_multiprocessing, client_config
    ) as engine:
        yield engine


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we inject custom routes and build async engine client differently
async def custom_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with custom_build_async_engine_client(args, client_config) as engine_client:
        app = build_app(args)

        ### CUSTOM ENDPOINTS ###
        @app.post("/reload_weights")
        async def _reload_weights(request: Request):
            data = await request.json()
            model_path = data.get("model_path")
            await engine_client.collective_rpc("custom_reload_weights", args=(model_path,))
            return {"status": "ok"}

        @app.post("/reset_weights")
        async def _reset_weights(request: Request):
            await engine_client.collective_rpc("custom_reset_weights")
            return {"status": "ok"}

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we call `custom_run_server_worker` instead of `run_server_worker`
async def custom_run_server(args: Namespace, **uvicorn_kwargs) -> None:
    listen_address, sock = setup_server(args)
    await custom_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference is that we call `custom_run_server` instead of `run_server` and we do config translation (i.e. pass populated namespace to `parse_args`)
def server(config: InferenceConfig, vllm_args: list[str]):
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)
    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(custom_run_server(args))
