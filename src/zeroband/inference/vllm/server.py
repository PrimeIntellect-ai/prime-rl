import signal
from argparse import Namespace

import uvloop
from fastapi import Request
from vllm.entrypoints.openai.api_server import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    FlexibleArgumentParser,
    UsageContext,
    build_app,
    create_server_socket,
    init_app_state,
    make_arg_parser,
    serve_http,
    set_ulimit,
    validate_parsed_serve_args,
)

from zeroband.inference.config import InferenceConfig


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "zeroband.inference.vllm.worker.CheckpointWorker"
    engine = AsyncLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)
    app = build_app(args)

    # Inject a custom endpoint
    @app.get("/v1/test_rpc")
    async def _test_rpc(request: Request):
        await engine.collective_rpc("test_rpc")
        return {"status": "ok"}

    @app.post("/v1/reload_weights")
    async def _reload_weights(request: Request):
        data = await request.json()
        model_path = data.get("model_path")
        await engine.collective_rpc("reload_weights", args=(model_path,))
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
    uvloop.run(run_server(args))
