import importlib

from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig

_MODEL_EXPORTS = {
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.gpt_oss.modeling_gpt_oss")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GptOssConfig",
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
]
