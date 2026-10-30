"""Custom SFT objectives (``[objective]``): a loss in place of token cross-entropy, with the rest of the SFT trainer
unchanged."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from jaxtyping import Float
from torch import Tensor, nn

from prime_rl.configs.sft import SFTObjectiveConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.sft.data import Batch
from prime_rl.utils.utils import import_object


@dataclass
class ObjectiveInputs:
    """Inputs for computing the objective on one micro-batch."""

    model: nn.Module
    hidden_states: Float[Tensor, "batch seq hidden"]
    """The backbone's final (post-norm) hidden states, without the vocabulary projection."""
    batch: Batch
    """The micro-batch on the device."""


@dataclass
class ObjectiveOutputs:
    """Outputs from computing the objective on one micro-batch."""

    loss: Float[Tensor, ""]
    """Summed loss. The trainer divides it by the global ``count`` over all ranks and micro-batches of the step."""
    count: int | Tensor
    """What the loss is averaged over (e.g. samples or tokens), for this micro-batch."""
    metrics: dict[str, Tensor] = field(default_factory=dict)
    """Summed metrics, divided like the loss and logged as ``loss/<name>`` in training and ``val/<name>`` in
    validation. Every call must return the same names."""


ObjectiveFn = Callable[..., ObjectiveOutputs]
"""A custom objective: ``my_loss(inputs: ObjectiveInputs, **kwargs) -> ObjectiveOutputs``, called with
``[objective.kwargs]``."""


def load_objective(config: SFTObjectiveConfig) -> Callable[[ObjectiveInputs], ObjectiveOutputs]:
    fn: ObjectiveFn = import_object(config.import_path)
    return partial(fn, **config.kwargs)


def objective_loss(
    objective: Callable[[ObjectiveInputs], ObjectiveOutputs], model: nn.Module, batch: Batch
) -> ObjectiveOutputs:
    """The objective's loss on one micro-batch, computed inside the model's forward pass on the backbone's final
    hidden states, where stock SFT computes its cross-entropy: FSDP keeps the root unit gathered for it and treats the
    modules it calls, such as the LM head, as part of the forward."""

    def loss(hidden_states: Tensor) -> dict:
        outputs = objective(ObjectiveInputs(model=model, hidden_states=hidden_states, batch=batch))
        # A dict, not the dataclass: FSDP hooks backward onto the tensors of the forward's output pytree.
        return {"loss": outputs.loss, "count": outputs.count, "metrics": outputs.metrics}

    outputs = forward(
        model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], hidden_states_fn=loss
    )
    return ObjectiveOutputs(**outputs)
