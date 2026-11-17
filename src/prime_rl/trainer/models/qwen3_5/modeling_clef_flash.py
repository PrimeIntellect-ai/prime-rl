"""Clef-Flash: a Qwen3.5 backbone with Clef's joint schema head, the model vLLM 0.33 serves as
``ClefFlashForJointSchema`` (``vllm/model_executor/models/clef_flash.py``)."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig

from prime_rl.trainer.models.conversion_ops import Drop, MapValue
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import JointSchemaHead
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

if TYPE_CHECKING:
    from prime_rl.trainer.sft.system_one.data import SystemOneStrategy

HEAD = "pooler.head."
SCALARS = ("prior_logit_scale", "joint_logit_scale", "residual_gate")
# Every name of the class's state dict starts with one of these; the release's head (joint_head.safetensors) has none.
PREFIXES = ("model.", "lm_head.", "mtp.", HEAD, "lexical_embedding.")


def to_vector(scalar: Tensor) -> Tensor:
    return scalar.reshape(1)


def to_scalar(vector: Tensor) -> Tensor:
    return vector.reshape(())


class ClefFlashForJointSchema(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 with Clef's joint schema head. The loader builds this class for a
    ``Qwen3_5ForConditionalGeneration`` checkpoint with a ``joint_head_config.json`` beside its config, which it
    attaches as ``config.joint_head_config``, as vLLM does. It adds two modules:

    ``pooler.head``        vLLM's ``JointSchemaHead``, under vLLM's names. FSDP refuses 0-dimensional parameters, so its
                           three scalars hold one element each; their HF weights are 0-dimensional, as in the release's
                           ``joint_head.safetensors``, which stores the head without the prefix.
    ``lexical_embedding``  a frozen copy of the LM head's rows, for the head's lexical prior, which vLLM takes from the
                           LM head. The release lacks it: ``init_missing_weights`` copies the loaded LM head. It is not
                           an HF weight.

    The LM head is frozen too: the loss never calls it, and the lexical copy stays equal to it. It must be untied."""

    export_weight_files = {HEAD: "joint_head.safetensors"}

    def __init__(self, config) -> None:
        name = type(self).__name__
        head_config = getattr(config, "joint_head_config", None)
        if not isinstance(head_config, dict):
            raise ValueError(f"{name} needs a joint_head_config.json beside the checkpoint's config.json")
        if getattr(config, "tie_word_embeddings", False):
            raise ValueError(f"{name} freezes the LM head, so it needs a backbone with an untied one")
        super().__init__(config)
        self.pooler = nn.Module()
        self.pooler.head = JointSchemaHead(**head_config)
        for scalar in SCALARS:
            setattr(self.pooler.head, scalar, nn.Parameter(torch.zeros(1)))
        self.lexical_embedding = nn.Embedding(self.lm_head.out_features, self.lm_head.in_features)
        self.lexical_embedding.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Whether it holds the release's head, which ``joint_head.safetensors`` stores without a prefix."""
        return "prior_logit_scale" in state_dict

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        added = any(name.startswith((HEAD, "lexical_embedding.")) for name in state_dict)
        return not cls.is_hf_state_dict(state_dict) and (added or super().is_prime_state_dict(state_dict))

    @classmethod
    def conversion_chain(cls, config):
        scalars = [MapValue(f"{HEAD}{name}", to_vector, to_scalar) for name in SCALARS]
        return [*super().conversion_chain(config), *scalars, Drop("lexical_embedding.weight")]

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """The release's head, which ``joint_head.safetensors`` stores without a prefix, moves under ``pooler.head``."""
        for name in [name for name in state_dict if not name.startswith(PREFIXES)]:
            state_dict[f"{HEAD}{name}"] = state_dict.pop(name)
        return super().convert_to_prime(state_dict)

    def init_missing_weights(self, names: list[str]) -> dict[str, Tensor]:
        """The lexical copy of the loaded LM head."""
        return {name: self.lm_head.weight for name in names if name == "lexical_embedding.weight"}

    def export_config_files(self, config: PretrainedConfig) -> dict[str, dict]:
        """``joint_head_config.json``, taken off `config`, so the exported ``config.json`` is the release's."""
        return {"joint_head_config.json": vars(config).pop("joint_head_config")}

    @classmethod
    def system_one_strategy(cls) -> type["SystemOneStrategy"]:
        """vLLM's read strategy for Clef-Flash, without its softmax: ``ClefFlashStrategy``."""
        from prime_rl.trainer.sft.system_one.strategies.clef import ClefFlashStrategy

        return ClefFlashStrategy


__all__ = ["ClefFlashForJointSchema"]
