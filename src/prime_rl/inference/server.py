import logging.config
import os

from prime_rl.configs.inference import InferenceConfig


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Router replay needs the V1 GPU model runner: V2 silently never captures
    # routed experts. Sampling-mask capture (sampling replay) conversely exists
    # only on V2 — config validation rejects enabling both. Everything else
    # follows vLLM's own V1/V2 resolution. setdefault so it stays overridable.
    if config.vllm.enable_return_routed_experts:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    if config.enable_return_sampling_mask:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

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
