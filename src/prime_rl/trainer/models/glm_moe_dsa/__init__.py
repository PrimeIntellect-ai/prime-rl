import importlib

from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

_MODEL_EXPORTS = {
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaModel",
    "GlmMoeDsaPreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaModel",
    "GlmMoeDsaPreTrainedModel",
]
