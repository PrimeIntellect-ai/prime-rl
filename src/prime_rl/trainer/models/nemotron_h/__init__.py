import importlib

from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

_MODEL_EXPORTS = {
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("prime_rl.trainer.models.nemotron_h.modeling_nemotron_h")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NemotronHConfig",
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]
