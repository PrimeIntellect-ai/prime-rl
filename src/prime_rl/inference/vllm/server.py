import signal
import traceback
from argparse import Namespace

import torch
import uvloop
from fastapi import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit

from prime_rl.inference.config import InferenceConfig


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
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
        try:
            data = await request.json()
            model_path = data.get("model_path")
            state_dict = torch.load(model_path, map_location="cpu", mmap=True)

            def weights_iterator():
                for key, value in state_dict.items():
                    if not key:
                        continue
                    yield key, value

            model = engine.engine.model_executor.driver_worker.model_runner.model
            model.load_weights(weights_iterator())

            # Process weights after loading (important for some models)
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model_config = engine.engine.model_config
            device = next(model.parameters()).device
            process_weights_after_loading(model, model_config, device)
            return {"status": "ok"}
        except Exception as e:
            print(f"Error reloading weights: {e}")
            print(f"Error reloading weights: {traceback.format_exc()}")
            raise e

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
