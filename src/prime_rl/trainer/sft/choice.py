"""Choice logits and loss inputs for ``packed_choice`` SFT data.

A position is supervised when its ``choice_ids`` row starts with a vocabulary id; the rest of the row
is right-padded with ``-1``. Every tensor here keeps the supervised rows in flattened sequence order.
"""

from typing import TypedDict

import torch
from torch import Tensor


def supervised_rows(choice_ids: Tensor) -> Tensor:
    """Boolean mask over the flattened positions of ``choice_ids`` ``[..., K]`` that carry choices."""
    return choice_ids.reshape(-1, choice_ids.shape[-1])[:, 0] >= 0


def choice_logits(hidden: Tensor, weight: Tensor, choice_ids: Tensor) -> Tensor:
    """``[M, K]`` fp32 logits of each supervised position's choice tokens, without full-vocabulary logits.

    ``hidden`` is ``[..., H]`` and ``choice_ids`` ``[..., K]`` over the same leading dims; ``weight`` is the
    ``[V, H]`` output embedding. Padded choice slots are 0.
    """
    rows = supervised_rows(choice_ids)
    ids = choice_ids.reshape(-1, choice_ids.shape[-1])[rows]
    row_hidden = hidden.reshape(-1, hidden.shape[-1])[rows]
    logits = torch.einsum("mh,mkh->mk", row_hidden, weight[ids.clamp_min(0)]).float()
    return logits.masked_fill(ids < 0, 0.0)


class ChoiceLossInputs(TypedDict):
    choice_logits: Tensor
    choice_counts: Tensor
    target_probs: Tensor
    weights: Tensor
    mix_weights: Tensor


def choice_loss_inputs(
    choice_logits: Tensor,
    choice_ids: Tensor,
    choice_targets: Tensor,
    choice_weights: Tensor,
    choice_mix_weights: Tensor,
) -> ChoiceLossInputs:
    """Keyword arguments for the ``[loss]`` function, taken at the rows ``choice_logits`` was computed for."""
    rows = supervised_rows(choice_ids)
    ids = choice_ids.reshape(-1, choice_ids.shape[-1])[rows]
    return {
        "choice_logits": choice_logits,
        "choice_counts": (ids >= 0).sum(dim=-1),
        "target_probs": choice_targets.reshape(-1, choice_targets.shape[-1])[rows],
        "weights": choice_weights.reshape(-1)[rows],
        "mix_weights": choice_mix_weights.reshape(-1, choice_mix_weights.shape[-1])[rows],
    }
