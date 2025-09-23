from collections.abc import Callable

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv

BackendLauncher = Callable[[InferenceConfig], None]


def _get_backend(server_type: str) -> BackendLauncher:
    """Return the backend launcher for the given server type.
    """

    if server_type == "vllm":
        from prime_rl.inference.backends.vllm import startup as launch_vllm

        return launch_vllm
    if server_type == "sglang":
        try:
            from prime_rl.inference.backends.sglang import startup as launch_sglang
        except ModuleNotFoundError as exc:
            raise RuntimeError("SGLang backend requested but not installed.") from exc
        return launch_sglang
    raise ValueError(f"Unsupported server type: {server_type}")


def main() -> None:
    config = parse_argv(InferenceConfig, allow_extras=True)
    server_type = config.server.type
    if server_type == "sglang":
        config.set_unknown_args(config.to_sglang() + config.get_unknown_args())
    launcher = _get_backend(server_type)
    launcher(config)


if __name__ == "__main__":
    main()
