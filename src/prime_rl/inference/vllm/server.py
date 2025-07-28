import signal
from argparse import Namespace

import uvloop
from fastapi import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
    run_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit

from prime_rl.inference.config import InferenceConfig


async def custom_run_server(args: Namespace) -> None:
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
    engine = AsyncLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)
    app = build_app(args)

    # Inject custom endpoint
    @app.post("/reload_weights")
    async def _reload_weights(request: Request):
        data = await request.json()
        model_path = data.get("model_path")
        await engine.collective_rpc("reload_weights", args=(model_path,))
        return {"status": "ok"}

    @app.post("/reset_weights")
    async def _reset_weights(request: Request):
        await engine.collective_rpc("reset_weights")
        return {"status": "ok"}

    vllm_config = await engine.get_vllm_config()
    await init_app_state(engine, vllm_config, app.state, args)

    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )

    await shutdown_task

    sock.close()


def server(config: InferenceConfig, vllm_args: list[str]):
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)

    # From vllm/entrypoints/cli/serve.py
    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
