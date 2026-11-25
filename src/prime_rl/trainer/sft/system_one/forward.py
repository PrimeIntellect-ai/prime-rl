"""The forward pass of a System One micro-batch."""

from functools import partial

import torch
from torch import nn

from prime_rl.trainer.model import forward
from prime_rl.trainer.sft.system_one.data import SystemOneStrategy
from prime_rl.trainer.sft.system_one.loss import DecisionLossOutput, SystemOneBatch, decision_loss


def decision_forward(
    model: nn.Module,
    micro_batch: SystemOneBatch,
    *,
    strategy: SystemOneStrategy,
    brier_weight: float,
    device: torch.device | str,
) -> DecisionLossOutput:
    """The micro-batch's decision loss, computed inside the model's forward on ``device``."""
    input_ids = micro_batch["input_ids"].to(device, non_blocking=True)
    prefix_lens = micro_batch["prefix_lens"]
    # Bound in a partial, so FSDP does not cast the float32 targets to bf16 as it does forward inputs
    loss_fn = partial(
        decision_loss,
        model,
        strategy=strategy,
        inputs=micro_batch["decision_loss_inputs"].to(device),
        input_ids=input_ids,
        brier_weight=brier_weight,
    )
    return forward(
        model,
        input_ids,
        micro_batch["position_ids"].to(device, non_blocking=True),
        seq_lens=micro_batch["seq_lens"].to(device, non_blocking=True),
        hidden_states_fn=loss_fn,
        prefix_lens=None if prefix_lens is None else prefix_lens.to(device, non_blocking=True),
    )
