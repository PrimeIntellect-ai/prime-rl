"""Inkling 2 (text tower) trainer implementation."""

from prime_rl.trainer.models.inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from prime_rl.trainer.models.inkling2.modeling_inkling2 import (
    Inkling2ForCausalLM,
    Inkling2Model,
    Inkling2PreTrainedModel,
)

__all__ = [
    "Inkling2Config",
    "Inkling2TextConfig",
    "Inkling2ForCausalLM",
    "Inkling2Model",
    "Inkling2PreTrainedModel",
]
