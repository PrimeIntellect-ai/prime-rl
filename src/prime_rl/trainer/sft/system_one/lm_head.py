"""Option logits read from the LM head, for strategies whose options are tokens."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor


def lm_head_logits(
    model: nn.Module, hidden_states: Tensor, positions: Sequence[int], token_ids: Sequence[Sequence[int]]
) -> Tensor:
    """The LM head's float32 logits of the tokens ``token_ids[i]`` at position ``positions[i]`` of the packed row's
    final ``hidden_states`` (``[seq, hidden]``), flat, in that order. One matmul of the read positions' hidden states
    with those tokens' rows of the LM head's weight, in float32 from the operands' own dtype, as the LM head computes
    its logits. The gradient reaches the hidden states read and only those rows of the weight.

    It reads ``model.lm_head.weight`` instead of calling the head, so it must run inside the forward, as
    ``SystemOneStrategy.option_logits`` does. FSDP holds the weight whole there: an untied LM head stays gathered with
    the final norm, a tied one with the root unit. A weight that is still a shard raises.
    """
    weight = model.lm_head.weight
    if isinstance(weight, DTensor):
        raise RuntimeError(
            "lm_head_logits read a shard of the LM head's weight: it must run inside the model's forward, where FSDP "
            "has gathered the weight"
        )
    assert len(positions) == len(token_ids), f"{len(positions)} positions for {len(token_ids)} lists of token ids"
    rows = sorted({token for tokens in token_ids for token in tokens})
    column = {token: index for index, token in enumerate(rows)}
    cells = [read * len(rows) + column[token] for read, tokens in enumerate(token_ids) for token in tokens]
    # One host-to-device copy for the three index vectors
    index = torch.tensor([*positions, *rows, *cells], device=hidden_states.device)
    position_index, row_index, cell_index = index.split([len(positions), len(rows), len(cells)])
    logits = hidden_states.index_select(0, position_index).float() @ weight.index_select(0, row_index).float().T
    softcap = getattr(model.lm_head, "softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits.take(cell_index)
