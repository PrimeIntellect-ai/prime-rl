import os

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    lp = config.logits_processors
    if lp.gibberish_detection is not None:
        gd = lp.gibberish_detection
        os.environ["PRIME_GIBBERISH_DETECTION_ENABLED"] = "1"
        os.environ["PRIME_GIBBERISH_DETECTION_TOKEN_ID_THRESHOLD"] = str(gd.token_id_threshold)
        os.environ["PRIME_GIBBERISH_DETECTION_LOGPROB_OFFSET"] = str(gd.logprob_offset)
    if lp.repetition_detection is not None:
        rd = lp.repetition_detection
        os.environ["PRIME_REPETITION_DETECTION_ENABLED"] = "1"
        os.environ["PRIME_REPETITION_DETECTION_WINDOW"] = str(rd.window)
        os.environ["PRIME_REPETITION_DETECTION_PROB_THRESHOLD"] = str(rd.prob_threshold)


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
