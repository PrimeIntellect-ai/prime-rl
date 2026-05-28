import importlib

from prime_rl.trainer.models.afmoe.configuration_afmoe import AfmoeConfig

_MODEL_EXPORTS = {
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.afmoe.modeling_afmoe")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AfmoeConfig",
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]
