import importlib

from prime_rl.trainer.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

_MODEL_EXPORTS = {
    "Glm4MoeForCausalLM",
    "Glm4MoeModel",
    "Glm4MoePreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.glm4_moe.modeling_glm4_moe")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Glm4MoeConfig",
    "Glm4MoeForCausalLM",
    "Glm4MoeModel",
    "Glm4MoePreTrainedModel",
]
