"""Custom SFT objectives (``[objective]``): the sample format and the loss, with the rest of the SFT trainer unchanged."""

import inspect
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor, nn

from prime_rl.configs.sft import SFTObjectiveConfig
from prime_rl.trainer.model import forward
from prime_rl.trainer.sft.data import Batch, Sample, SampleInputs
from prime_rl.utils.utils import import_object


@dataclass
class ObjectiveInputs:
    """Inputs for computing the objective on one micro-batch."""

    model: nn.Module
    hidden_states: Float[Tensor, "batch seq hidden"]
    """The backbone's final (post-norm) hidden states, without the vocabulary projection."""
    batch: Batch
    """The micro-batch on the device, including ``objective_data`` when the samples carry it."""


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


@runtime_checkable
class SFTObjective(Protocol):
    def make_sample(self, inputs: SampleInputs) -> Sample | None:
        """Builds one sample from a dataset row, or None to skip it.

        Shuffling, rank sharding, epochs, resume and packing stay the trainer's. ``input_ids``, ``position_ids``,
        ``loss_mask`` and ``target_ids`` are lists of one length; ``seq_lens`` may be omitted (packing records one
        entry per sample, so a sample holding several independent sequences must keep them apart itself, for example
        through a model class that derives boundaries from ``position_ids`` restarting at 0). Samples longer than
        ``seq_len`` are skipped. An optional ``objective_data`` dict of plain values carries what the loss needs
        beyond tokens (e.g. targets) to ``batch["objective_data"]``: one dict per sample of the micro-batch, in order.
        Runs in the dataloader workers: the objective must be picklable.
        """
        ...

    def loss(self, inputs: ObjectiveInputs) -> ObjectiveOutputs: ...


def _holds_parameters(objective: object) -> bool:
    """Whether the objective, or a module or parameter it holds directly, has parameters."""
    for value in [objective, *getattr(objective, "__dict__", {}).values()]:
        if isinstance(value, nn.Parameter) or (
            isinstance(value, nn.Module) and next(value.parameters(), None) is not None
        ):
            return True
    return False


def load_objective(config: SFTObjectiveConfig) -> SFTObjective:
    objective = import_object(config.import_path)(**config.kwargs)
    if _holds_parameters(objective):
        # The optimizer, rank sync and checkpoints see only the model's parameters.
        raise TypeError(f"{config.import_path} holds parameters, which would not be trained; give them to the model")
    if not isinstance(objective, SFTObjective):
        raise TypeError(f"{config.import_path} must implement make_sample and loss (see SFTObjective)")
    for name in ("make_sample", "loss"):
        # A Protocol checks names, not signatures; a wrong arity would otherwise fail inside a dataloader worker.
        try:
            inspect.signature(getattr(objective, name)).bind(None)
        except TypeError:
            raise TypeError(
                f"{config.import_path}.{name} must accept one positional argument (see SFTObjective)"
            ) from None
    return objective


def objective_loss(objective: SFTObjective, model: nn.Module, batch: Batch) -> ObjectiveOutputs:
    """The objective's loss on one micro-batch, computed inside the model's forward pass on the backbone's final
    hidden states, where stock SFT computes its cross-entropy: FSDP keeps the root unit gathered for it and treats the
    modules it calls, such as the LM head, as part of the forward."""

    def loss(hidden_states: Tensor) -> dict:
        outputs = objective.loss(ObjectiveInputs(model=model, hidden_states=hidden_states, batch=batch))
        # A dict, not the dataclass: FSDP hooks backward onto the tensors of the forward's output pytree.
        return {"loss": outputs.loss, "count": outputs.count, "metrics": outputs.metrics}

    outputs = forward(
        model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], hidden_states_fn=loss
    )
    return ObjectiveOutputs(**outputs)
