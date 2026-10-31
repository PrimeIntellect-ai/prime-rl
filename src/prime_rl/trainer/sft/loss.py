"""Custom SFT losses (``[loss]``): a loss function in place of token cross-entropy, with the rest of the SFT trainer
unchanged."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from jaxtyping import Float
from torch import Tensor, nn

from prime_rl.configs.sft import SFTCustomLossConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.sft.data import Batch
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing the loss on one micro-batch."""

    model: nn.Module
    hidden_states: Float[Tensor, "batch seq hidden"]
    """The backbone's final (post-norm) hidden states, without the vocabulary projection."""
    batch: Batch
    """The micro-batch on the device."""


@dataclass
class LossOutputs:
    """Outputs from computing the loss on one micro-batch."""

    loss: Float[Tensor, ""]
    """Summed loss. The trainer divides it by the global ``count`` over all ranks and micro-batches of the step."""
    count: int | Tensor
    """What the loss is averaged over (e.g. samples or tokens), for this micro-batch."""
    metrics: dict[str, Tensor] = field(default_factory=dict)
    """Summed metrics, divided like the loss and logged as ``loss/<name>`` in training and ``val/<name>`` in
    validation. Every call must return the same names."""


LossFn = Callable[..., LossOutputs]
"""A custom loss: ``my_loss(inputs: LossInputs, **kwargs) -> LossOutputs``, called with ``[loss.kwargs]``."""


def setup_sft_loss_fn(config: SFTCustomLossConfig) -> Callable[[LossInputs], LossOutputs]:
    fn: LossFn = import_object(config.import_path)
    return partial(fn, **config.kwargs)


def run_loss_fn(
    loss_fn: Callable[[LossInputs], LossOutputs],
    model: nn.Module,
    batch: Batch,
    *,
    seq_lens_are_pre_shard: bool = False,
) -> LossOutputs:
    """The custom loss on one micro-batch, computed inside the model's forward pass on the backbone's final
    hidden states, where stock SFT computes its cross-entropy: FSDP keeps the root unit gathered for it and treats the
    modules it calls, such as the LM head, as part of the forward."""

    def loss(hidden_states: Tensor) -> dict:
        outputs = loss_fn(LossInputs(model=model, hidden_states=hidden_states, batch=batch))
        # A dict, not the dataclass: FSDP hooks backward onto the tensors of the forward's output pytree.
        return {"loss": outputs.loss, "count": outputs.count, "metrics": outputs.metrics}

    outputs = forward(
        model,
        batch["input_ids"],
        batch["position_ids"],
        seq_lens=batch["seq_lens"],
        seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        hidden_states_fn=loss,
    )
    return LossOutputs(**outputs)
