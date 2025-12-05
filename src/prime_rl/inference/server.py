from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
