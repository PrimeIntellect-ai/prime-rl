from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def _get_backend_cls(server_type: str):
    """Return the backend class for the given server type.
    Imports are done lazily so optional backends (e.g., sglang) do not cause
    import-time failures when the dependency isn't installed.
    """
    if server_type == "vllm":
        from prime_rl.inference.backends.vllm import VLLMBackend

        return VLLMBackend
    if server_type == "sglang":
        try:
            from prime_rl.inference.backends.sglang import SGLangBackend
        except ModuleNotFoundError as e:
            raise RuntimeError("SGLang backend requested but not installed.") from e
        return SGLangBackend


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    server_type = config.server.server_type
    # Preserve sglang-specific argument transformation from upstream
    if server_type == "sglang":
        config.set_unknown_args(config.to_sglang() + config.get_unknown_args())
    backend_cls = _get_backend_cls(server_type)
    if backend_cls is None:
        raise ValueError(f"Unsupported server type: {server_type}")
    backend = backend_cls()
    backend.startup(config)


if __name__ == "__main__":
    main()
