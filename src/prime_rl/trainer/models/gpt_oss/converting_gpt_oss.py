"""State-dict conversion for GPT-OSS.

The custom prime-rl GPT-OSS implementation mirrors HuggingFace's parameter naming
exactly (gate_up_proj/gate_up_proj_bias/down_proj/down_proj_bias plus router.weight
and router.bias as nn.Parameters). So loading the unsloth BF16 checkpoint requires
no key conversion - HF and prime formats are identical for this model.
"""

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.experts.gate_up_proj" in name for name in state_dict.keys())


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return is_hf_state_dict(state_dict)


def conversion_chain(config) -> list[ConvOp]:
    # HF and prime layouts coincide for GPT-OSS, so the conversion is the identity.
    return []
