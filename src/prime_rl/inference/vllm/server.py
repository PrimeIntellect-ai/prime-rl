import sys
from typing import Any, Callable

import uvloop
from fastapi import Request

from prime_rl.inference.config import InferenceConfig
from vllm.config import LogprobsMode
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

# Platform check: this module relies on fork() semantics for multi-server mode
if not sys.platform.startswith("linux"):
    raise SystemExit("[prime-rl] Linux-only: this patch relies on fork() semantics")


# Apply patches at module import time to inject custom endpoints
def _apply_patches() -> None:
    """Apply monkey patches to vLLM API server for custom endpoints and worker config."""
    import vllm.entrypoints.openai.api_server as api_mod

    # Patch build_app to inject custom endpoints
    _orig_build_app: Callable = api_mod.build_app

    def _patched_build_app(args):
        app = _orig_build_app(args)

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            engine = api_mod.engine_client(request)
            data = await request.json()
            await engine.collective_rpc("update_weights", args=((data or {}).get("weight_dir"),))
            return {"status": "ok"}

        @app.post("/reload_weights")
        async def _reload_weights(request: Request):
            engine = api_mod.engine_client(request)
            await engine.collective_rpc("reload_weights")
            return {"status": "ok"}

        return app

    api_mod.build_app = _patched_build_app

    # Patch build_async_engine_client_from_engine_args to inject worker extension and logprobs config
    _orig_build_engine_client = api_mod.build_async_engine_client_from_engine_args

    def _patched_build_engine_client(engine_args, **kw: Any):
        engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
        engine_args.logprobs_mode = LogprobsMode.PROCESSED_LOGPROBS
        return _orig_build_engine_client(engine_args, **kw)

    api_mod.build_async_engine_client_from_engine_args = _patched_build_engine_client


# Apply patches immediately when this module is imported
_apply_patches()


def server(config: InferenceConfig, vllm_args: list[str]):
    """
    Start vLLM API server with custom patches.

    Custom functionality (worker extension, custom endpoints) is injected via
    monkey-patching applied at module import time.
    """
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)

    # Raise error if logprobs_mode is not set to processed_logprobs
    if args.logprobs_mode != "processed_logprobs":
        raise ValueError("logprobs_mode must be 'processed_logprobs' to be compatible with the orchestrator.")

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            # Force fork so child processes inherit the monkey patches
            # (spawn would create fresh interpreters without patches)
            import multiprocessing as mp

            try:
                if mp.get_start_method(allow_none=True) != "fork":
                    mp.set_start_method("fork")
            except RuntimeError:
                # Start method already set; ignore
                pass

            run_multi_api_server(args)
        else:
            # Single API server (this process)
            uvloop.run(run_server(args))
