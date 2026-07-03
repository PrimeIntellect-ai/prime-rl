import logging.config
import os

from prime_rl.configs.inference import InferenceConfig


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Force the V1 GPU model runner. vLLM 0.24.0 routes Llama/Mistral/Qwen3 plus MoE archs
    # (DeepseekV2, Qwen2Moe, GraniteMoe; the 0.23 MoE/quantized guard was removed) to the
    # V2 runner, which has no `_preprocess` hook for our padded-input scrub (see
    # inference/vllm/padded_input_scrub.py) and doesn't zero the padded decode tail
    # itself; that stale tail poisons CUDA-graph replay as NaN logits (reproduced on
    # Nemotron-Nano SWE, PR #2506; upstream fix vllm#42779 is still unmerged). V1 keeps
    # the scrub effective for every model and is also required for routed-experts capture
    # (the NIXL PD path rejects V2). setdefault so it stays overridable.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    # vLLM 0.24.0 flipped VLLM_ENFORCE_STRICT_TOOL_CALLING's default to True, which
    # grammar-constrains generation (xgrammar structural tags) for tool_choice
    # "required"/named and strict tools — a sampling distribution the trainer never
    # sees. Keep it off so rollout logprobs stay faithful for importance ratios.
    os.environ.setdefault("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")

    deep_gemm_enabled = "1" if config.use_deep_gemm else "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = deep_gemm_enabled
    os.environ["VLLM_MOE_USE_DEEP_GEMM"] = deep_gemm_enabled

    if config.enable_lora:
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
