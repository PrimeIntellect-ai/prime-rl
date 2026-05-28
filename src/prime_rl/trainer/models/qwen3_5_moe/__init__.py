import importlib

from prime_rl.trainer.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

_MODEL_EXPORTS = {
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
]
