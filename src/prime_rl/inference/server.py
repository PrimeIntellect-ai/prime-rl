import logging.config
import os

from prime_rl.configs.inference import InferenceConfig


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # vLLM captures sampling masks and routed experts only on the V2 model runner.
    # setdefault keeps an explicit env-var choice authoritative.
    if config.enable_return_sampling_mask or config.vllm.enable_return_routed_experts:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

    top_p_first = config.top_p_before_top_k and config.enable_return_sampling_mask
    if top_p_first:
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
            raise ValueError("top_p_before_top_k requires VLLM_USE_V2_MODEL_RUNNER=1")
    os.environ["PRL_TOP_P_BEFORE_TOP_K"] = "1" if top_p_first else "0"

    # vLLM 0.24.0 flipped VLLM_ENFORCE_STRICT_TOOL_CALLING's default to True, which
    # grammar-constrains generation (xgrammar structural tags) for tool_choice
    # "required"/named and strict tools — a sampling distribution the trainer never
    # sees. Keep it off so rollout logprobs stay faithful for importance ratios.
    os.environ.setdefault("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")

    deep_gemm_enabled = "1" if config.use_deep_gemm else "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = deep_gemm_enabled
    os.environ["VLLM_MOE_USE_DEEP_GEMM"] = deep_gemm_enabled

    if config.vllm.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if config.log.json_logging:
        # Route vLLM's stdlib loggers through a JSON formatter matching
        # trainer / orchestrator. The env var (not in-process dictConfig)
        # is what reaches vLLM's spawned workers.
        from prime_rl.inference.json_logging import build_dict_config, write_logging_config

        config_path = write_logging_config(config.log.level)
        # vLLM raises if VLLM_LOGGING_CONFIG_PATH is set while
        # VLLM_CONFIGURE_LOGGING=0 (its supported way to disable logger
        # setup). Force it on — opting into JSON logging is an explicit
        # request to configure vLLM's logger.
        os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
        os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(config_path)
        logging.config.dictConfig(build_dict_config(config.log.level))


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
    "modelexpress": "prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker",
}


# Adapted from vllm/entrypoints/cli/serve.py: translates the config into vLLM's parsed args. Everything
# prime-rl adds to the server comes from the prime-rl-vllm plugin, which vLLM loads on its own.
def server(config: InferenceConfig):
    import uvloop
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from prime_rl.inference.vllm.api_server import (
        LIVENESS_TIMEOUT_ENV,
        WEIGHT_PULL_SERVER_ENV,
        WEIGHT_PULL_STAGING_BUFFER_BYTES_ENV,
        WEIGHT_PULL_STAGING_BUFFERS_COUNT_ENV,
    )

    # Signal worker processes to disable LoRA on MoE layers when LoRA targets don't include experts
    lora_target_modules = config.vllm.lora_target_modules
    if lora_target_modules and not any("expert" in m for m in lora_target_modules):
        os.environ["PRIME_NO_MOE_LORA"] = "1"
    os.environ[LIVENESS_TIMEOUT_ENV] = str(config.server.liveness_timeout_seconds)

    namespace = config.to_namespace()

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
    assert args is not None
    validate_parsed_serve_args(args)

    args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]

    if (pull := config.weight_broadcast.pull) is not None:
        os.environ[WEIGHT_PULL_SERVER_ENV] = pull.server
        os.environ[WEIGHT_PULL_STAGING_BUFFER_BYTES_ENV] = str(pull.staging_buffer_bytes)
        os.environ[WEIGHT_PULL_STAGING_BUFFERS_COUNT_ENV] = str(pull.staging_buffers_count)
        # One engine at a time reads a version from the trainer and each engine feeds one peer at a time, so a
        # version spreads through the fleet in a doubling tree instead of every engine sharing the trainer's
        # network. Set before vLLM starts its workers.
        os.environ.setdefault("MX_MAX_TRAINER_READERS", "1")
        os.environ.setdefault("MX_MAX_TENSOR_READERS", "1")
        # An install keeps generation paused and holds its source, so a copy whose source stopped answering must
        # fail, shutting the server down, instead of waiting ModelExpress's default 900s on each staged batch. A
        # batch is at most one staging buffer, seconds of work on any working path.
        os.environ.setdefault("MX_TRANSFER_TIMEOUT", "120")
        # An engine serves the versions it installs to its peers only when ModelExpress loaded its weights.
        if args.load_format == "auto":
            args.load_format = "modelexpress"
            os.environ.setdefault("MX_LOAD_STRATEGY_CHAIN", "RL")
            os.environ.setdefault("MX_SERVER_ADDRESS", pull.server)

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    elif args.api_server_count > 1:
        run_multi_api_server(args)
    else:
        uvloop.run(run_server(args))
