import os

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)

    # Enable vLLM logging configuration if request logging is enabled
    # This ensures RequestLogger output goes to stdout/stderr
    if config.enable_log_requests:
        os.environ["VLLM_CONFIGURE_LOGGING"] = "1"

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
