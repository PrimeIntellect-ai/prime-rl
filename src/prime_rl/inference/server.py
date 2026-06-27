import logging.config
import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def _env_bool(value: bool) -> str:
    return "1" if value else "0"


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    deep_gemm_enabled = "1" if config.use_deep_gemm else "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = deep_gemm_enabled
    os.environ["VLLM_MOE_USE_DEEP_GEMM"] = deep_gemm_enabled

    sampler = config.finite_topk_sampled_logprob
    if sampler.enabled is not None:
        os.environ["PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB"] = _env_bool(sampler.enabled)
        if sampler.enabled:
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL"] = sampler.tail
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE"] = _env_bool(sampler.dense_presence)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL"] = str(sampler.stats_interval)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT"] = str(sampler.hit_log_limit)
            os.environ["PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK"] = _env_bool(sampler.log_fallback)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL"] = _env_bool(sampler.precompile_tail)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_K"] = str(sampler.precompile_top_k)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P"] = str(sampler.precompile_top_p)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_VOCAB"] = str(sampler.precompile_vocab)
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES"] = ",".join(
                str(batch) for batch in sampler.precompile_batches
            )
            os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_BOUNDARY_TIE_GUARD"] = _env_bool(
                sampler.boundary_tie_guard
            )

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


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
