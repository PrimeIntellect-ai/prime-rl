import importlib

from prime_rl.trainer.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

_MODEL_EXPORTS = {
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.gemma4.modeling_gemma4")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
]
