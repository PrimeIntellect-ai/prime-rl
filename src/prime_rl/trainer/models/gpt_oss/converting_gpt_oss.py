"""State-dict conversion for GPT-OSS.

The prime-rl GPT-OSS state-dict surface mirrors HuggingFace's parameter names and
layouts, so loading the BF16 checkpoint requires no key conversion.
"""

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.experts.gate_up_proj" in name for name in state_dict.keys())


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    # Prime format equals HF format for GPT-OSS, so we never claim to be a separate
    # prime format - this disables the auto-conversion path in load_dcp_from_hf and
    # lets DCP load HF safetensors directly.
    return False


def conversion_chain(config) -> list[ConvOp]:
    # HF and prime layouts coincide for GPT-OSS, so the conversion is the identity.
    return []
