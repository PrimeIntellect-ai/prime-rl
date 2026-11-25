"""Looped Inkling 2 (Inkling 2 Mini) for RL on a vLLM build that ships the model: vLLM's own class with BF16 experts
and weight updates loaded in place. Registered by `prime_rl.inference.patches.register_inkling2_loop_for_rl`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Inkling2LoopForRL

__all__ = ["Inkling2LoopForRL"]


def __getattr__(name):
    if name in __all__:
        # Imported on demand: the model needs `vllm.models.inkling2`, which the pinned vLLM release does not have.
        from . import model

        return model.Inkling2LoopForRL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
