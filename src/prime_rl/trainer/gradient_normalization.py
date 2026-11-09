from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GradientNormalization:
    """Per-component loss weights and one post-backward gradient scale that divide each component by its global count."""

    loss_weights: dict[str, float]
    grad_scale: float
    reference: float
    placement: Literal["grads", "loss"]

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, float],
        fsdp_divide_factor: float,
        placement: Literal["grads", "loss"],
    ) -> "GradientNormalization":
        """Build factors with ``grad_scale * sum_c loss_weights[c] * L_c == fsdp_divide_factor * sum_c L_c / S_c``."""
        counts = {name: count if count > 0 else 1.0 for name, count in counts.items()}
        reference = max(counts.values())
        if placement == "grads":
            loss_weights = {name: reference / count for name, count in counts.items()}
            grad_scale = fsdp_divide_factor / reference
        elif placement == "loss":
            loss_weights = {name: 1.0 / count for name, count in counts.items()}
            grad_scale = fsdp_divide_factor
        else:
            raise ValueError(f"Unknown gradient normalization placement: {placement}")
        return cls(loss_weights=loss_weights, grad_scale=grad_scale, reference=reference, placement=placement)

    def logged_loss(self, backward_loss: torch.Tensor) -> torch.Tensor:
        """Return the normalized loss ``sum_c L_c / S_c`` given the loss that was backpropagated."""
        return backward_loss / self.reference if self.placement == "grads" else backward_loss


def all_reduce_counts(
    local_counts: dict[str, float],
    group: dist.ProcessGroup,
    *,
    cp_gathered: bool,
    cp_size: int,
) -> dict[str, float]:
    """Sum per-component loss-token counts over ``group`` (dp_cp) in one fp64 all-reduce.

    Every CP rank counts the unsharded mask, so the sum is ``N * cp_size``. When the loss is
    computed on CP-gathered logprobs (``cp_gathered=True``, RL), every CP rank computes the whole
    sequence's loss and the gather's backward (a reduce-scatter sum) adds the CP ranks' gradients,
    so the ``cp_size`` in the count cancels and the count is kept as is. When each CP rank computes
    the loss on its own shard (``cp_gathered=False``, supervised), the count is divided by ``cp_size``.
    """
    counts = torch.tensor(list(local_counts.values()), dtype=torch.float64, device="cuda")
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    if not cp_gathered:
        counts /= cp_size
    return dict(zip(local_counts, counts.tolist(), strict=True))
