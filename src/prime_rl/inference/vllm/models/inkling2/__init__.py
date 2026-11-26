# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 (text) for vLLM on Hopper and Blackwell, registered out-of-tree."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Inkling2ForCausalLM

__all__ = ["Inkling2ForCausalLM", "register"]


def __getattr__(name):
    if name in __all__:
        from . import model, plugin

        return model.Inkling2ForCausalLM if name == "Inkling2ForCausalLM" else plugin.register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
