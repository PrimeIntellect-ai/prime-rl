"""State-dict conversion for GPT-OSS.

The custom prime-rl GPT-OSS implementation mirrors HuggingFace's parameter naming
exactly (gate_up_proj/gate_up_proj_bias/down_proj/down_proj_bias plus router.weight
and router.bias as nn.Parameters). So loading the unsloth BF16 checkpoint requires
no key conversion - HF and prime formats are identical for this model.
"""

from torch import Tensor


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.experts.gate_up_proj" in name for name in state_dict.keys())


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    # Prime format equals HF format for GPT-OSS, so we never claim to be a separate
    # prime format - this disables the auto-conversion path in load_dcp_from_hf and
    # lets DCP load HF safetensors directly.
    return False


def convert_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return state_dict


def convert_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return state_dict
