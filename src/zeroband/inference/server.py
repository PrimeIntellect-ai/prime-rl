from zeroband.inference.config import InferenceConfig
from zeroband.inference.vllm.server import server
from zeroband.utils.pydantic_config import parse_argv


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
