"""Custom SFT objectives (``[objective]``): the sample format and the loss, with the rest of the SFT trainer unchanged."""

import inspect
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor, nn

from prime_rl.configs.sft import SFTObjectiveConfig
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


def load_objective(config: SFTObjectiveConfig) -> SFTObjective:
    objective = import_object(config.import_path)(**config.kwargs)
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
