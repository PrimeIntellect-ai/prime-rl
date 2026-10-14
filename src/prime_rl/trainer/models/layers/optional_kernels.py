"""Kernels from the linux-gated `gpu` extra that layers bind at construction.

Without the extra the layers still build (the orchestrator builds the model on the meta device to
count FLOPs) and raise when run.
"""

import torch
from torch import nn


def unavailable(package: str):
    def raise_import_error(*args, **kwargs):
        raise ImportError(f"{package} is not installed; install the `gpu` extra")

    return raise_import_error


try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.conv import causal_conv1d
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.utils import prepare_sequence_ids
except ImportError:
    causal_conv1d = build_cp_context = chunk_gated_delta_rule = prepare_sequence_ids = unavailable(
        "flash-linear-attention"
    )

    class FusedRMSNormGated(nn.Module):
        """Holds the weight of fla's `FusedRMSNormGated`."""

        def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "swish"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))

        forward = unavailable("flash-linear-attention")
