import importlib

from prime_rl.trainer.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

_MODEL_EXPORTS = {
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2Model",
    "MiniMaxM2PreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.minimax_m2.modeling_minimax_m2")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2Model",
    "MiniMaxM2PreTrainedModel",
]
