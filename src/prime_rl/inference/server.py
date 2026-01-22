from pydantic_config import cli

from prime_rl.inference.config import InferenceConfig


def main():
    config = cli(InferenceConfig, allow_extras=True)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
