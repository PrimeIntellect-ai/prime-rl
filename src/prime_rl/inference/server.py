import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def setup_inference_env(config: InferenceConfig):
    if config.backend in ("vllm", "dynamo"):
        setup_vllm_env(config)


def main():
    config = cli(InferenceConfig)
    setup_inference_env(config)

    if config.backend == "vllm":
        # We import here to be able to set environment variables before importing vLLM
        from prime_rl.inference.vllm.server import server  # pyright: ignore

        server(config, vllm_extra=config.vllm_extra)
    elif config.backend == "sglang":
        from prime_rl.inference.sglang.server import server

        server(config)
    elif config.backend == "dynamo":
        from prime_rl.inference.dynamo.server import server

        server(config)
    else:
        raise ValueError(f"Unsupported inference backend: {config.backend}")


if __name__ == "__main__":
    main()
