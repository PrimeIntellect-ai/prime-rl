import os
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    compat_site = Path(__file__).with_name("compat_site")
    pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = f"{compat_site}{os.pathsep}{pythonpath}" if pythonpath else str(compat_site)

    # Also patch this interpreter; the PYTHONPATH hook above covers vLLM's
    # spawned EngineCore processes.
    from transformers.models import qwen2_vl
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor

    qwen2_vl.__dict__.setdefault("Qwen2VLImageProcessorFast", Qwen2VLImageProcessor)

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["VLLM_USE_DEEP_GEMM"] = "1" if config.use_deep_gemm else "0"
    os.environ["VLLM_MOE_USE_DEEP_GEMM"] = "1" if config.use_deep_gemm else "0"

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if config.use_deep_gemm:
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
