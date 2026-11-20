from torch import Tensor, nn
from torch.nn.utils import parametrize

from prime_rl.trainer.models.conversion_ops import MapValue, Rename
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import JOINT_HEAD_WEIGHTS, JointSchemaHead
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

HEAD = "pooler.head."
SCALARS = ("prior_logit_scale", "joint_logit_scale", "residual_gate")


def prime_name(scalar: str) -> str:
    """Where the class holds one of the head's scalars."""
    return f"{HEAD}parametrizations.{scalar}.original"


def to_vector(scalar: Tensor) -> Tensor:
    return scalar.reshape(1)


def to_scalar(vector: Tensor) -> Tensor:
    return vector.reshape(())


class OneElement(nn.Module):
    """A 0-dimensional parameter held as one element, as FSDP refuses 0-dimensional parameters."""

    def forward(self, vector: Tensor) -> Tensor:
        return to_scalar(vector)

    def right_inverse(self, scalar: Tensor) -> Tensor:
        return to_vector(scalar)


def check_joint_head_config(config) -> None:
    """Raise if ``config`` is tied or its ``joint_head_config`` is for another backbone width."""
    if getattr(config, "tie_word_embeddings", False):
        raise ValueError("ClefFlashForJointSchema freezes its LM head, so it must be untied from the input embedding")
    # The head's own constructor checks the rest
    head_size = config.joint_head_config.get("hidden_size")
    hidden_size = getattr(config, "text_config", config).hidden_size
    if head_size is not None and head_size != hidden_size:
        raise ValueError(f"joint_head_config.json has hidden_size {head_size}, but the backbone's is {hidden_size}")


class ClefFlashForJointSchema(Qwen3_5ForCausalLM):
    """Qwen3.5 with Clef's joint schema head at ``pooler.head``, whose lexical prior reads the frozen LM head's rows."""

    extra_config_file = "joint_head_config.json"
    extra_config_attr = "joint_head_config"
    stock_model_type = "qwen3_5"
    stock_architecture = "Qwen3_5ForConditionalGeneration"
    export_files = {HEAD: JOINT_HEAD_WEIGHTS}

    def __init__(self, config) -> None:
        check_joint_head_config(config)
        super().__init__(config)
        self.pooler = nn.Module()
        self.pooler.head = JointSchemaHead(**config.joint_head_config)
        for scalar in SCALARS:
            parametrize.register_parametrization(self.pooler.head, scalar, OneElement())
        self.lm_head.weight.requires_grad_(False)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return f"{HEAD}{SCALARS[0]}" in state_dict

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return prime_name(SCALARS[0]) in state_dict

    @classmethod
    def conversion_chain(cls, config):
        scalars = [
            op
            for scalar in SCALARS
            for op in (
                Rename(f"{HEAD}{scalar}", prime_name(scalar)),
                MapValue(prime_name(scalar), to_vector, to_scalar),
            )
        ]
        return [*super().conversion_chain(config), *scalars]


__all__ = ["ClefFlashForJointSchema"]
