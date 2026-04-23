"""Weight key conversions between HF and prime-rl for Gemma4.

Our prime-rl Gemma4 mirrors HF's weight layout (same param names under
``model.layers.{i}.*``, packed ``experts.gate_up_proj`` / ``experts.down_proj``,
per-head q/k/v norms, etc.). So conversion is a no-op — both sides agree.

These stubs exist for symmetry with other model ports; future customizations
(e.g. switching experts to a GroupedExperts-style layout with w1/w2/w3) would
land here.
"""

from __future__ import annotations

from torch import Tensor


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return state_dict


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return state_dict


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    return state_dict


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    return state_dict
