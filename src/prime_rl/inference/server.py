import logging.config
import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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
