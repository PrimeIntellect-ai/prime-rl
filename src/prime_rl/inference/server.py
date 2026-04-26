import os
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def resolve_env_value(value: str, project_dir: Path) -> str:
    resolved = value.replace("$PROJECT_DIR", str(project_dir))
    if "${LD_LIBRARY_PATH:-}" in resolved:
        resolved = resolved.replace("${LD_LIBRARY_PATH:-}", os.environ.get("LD_LIBRARY_PATH", ""))
    return resolved


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    for key, value in config.env_overrides.items():
        os.environ[key] = resolve_env_value(str(value), Path.cwd())

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
