import importlib

from prime_rl.trainer.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

_MODEL_EXPORTS = {
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.qwen3_moe.modeling_qwen3_moe")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
]
