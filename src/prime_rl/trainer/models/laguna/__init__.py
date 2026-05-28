import importlib

from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig

_MODEL_EXPORTS = {
    "LagunaForCausalLM",
    "LagunaModel",
    "LagunaPreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.laguna.modeling_laguna")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LagunaConfig",
    "LagunaForCausalLM",
    "LagunaModel",
    "LagunaPreTrainedModel",
]
