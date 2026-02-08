import os

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
