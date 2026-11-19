"""The decision loss (``[loss] type = "decision"``)."""

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from prime_rl.trainer.data import Batch, Sample, cat_collate
from prime_rl.trainer.sft.system_one.data import ReadQuestion, SystemOneStrategy


@dataclass(frozen=True)
class DecisionLossInputs:
    """A micro-batch's decision-loss inputs, built on the host from its ``sample_lens`` and ``sample_data``."""

    input_ids: Int[Tensor, "seq"]
    """The row's token ids."""
    questions: list[ReadQuestion]
    """The micro-batch's questions, with their reads and records in row positions."""
    option_slots: Int[Tensor, "options"]
    """Each option's flat index in a ``[questions, max_options]`` grid."""
    labelled: Int[Tensor, "labelled"]
    """Indices into ``questions`` of the questions with a target."""
    targets: Float[Tensor, "labelled max_options"]
    """Targets of the labelled questions (float32), zero-padded to the most options."""

    @property
    def num_labelled(self) -> int:
        return self.labelled.numel()

    def _map(self, fn) -> "DecisionLossInputs":
        tensors = ("input_ids", "option_slots", "labelled", "targets")
        return replace(self, **{name: fn(getattr(self, name)) for name in tensors})

    def pin_memory(self) -> "DecisionLossInputs":
        """Copies in pinned memory, called by the dataloader."""
        return self._map(Tensor.pin_memory)

    def to(self, device: torch.device | str) -> "DecisionLossInputs":
        """Copies on ``device``."""
        return self._map(lambda tensor: tensor.to(device, non_blocking=True))


def build_decision_loss_inputs(
    input_ids: Int[Tensor, "seq"], sample_lens: list[int], sample_data: list[dict]
) -> DecisionLossInputs:
    """The decision-loss inputs of one packed row."""
    questions, labelled, targets = [], [], []
    offset = 0
    for sample_len, record in zip(sample_lens, sample_data, strict=True):
        for question in record["questions"]:
            reads = []
            for start, end in question["reads"]:
                assert end <= sample_len, f"record {record['id']!r} has {sample_len} tokens, read up to {end}"
                reads.append((offset + start, offset + end))
            if question["target"] is not None:
                labelled.append(len(questions))
                targets.append(question["target"])
            questions.append(
                ReadQuestion(
                    record_id=record["id"],
                    id=question["id"],
                    type=question["type"],
                    options=tuple(question["options"]),
                    reads=tuple(reads),
                    record=(offset, offset + sample_len),
                )
            )
        offset += sample_len
    width = max(len(question.options) for question in questions)
    option_slots = [index * width + option for index, q in enumerate(questions) for option in range(len(q.options))]
    return DecisionLossInputs(
        input_ids=input_ids,
        questions=questions,
        option_slots=torch.tensor(option_slots, dtype=torch.long),
        labelled=torch.tensor(labelled, dtype=torch.long),
        targets=torch.tensor(
            [target + [0.0] * (width - len(target)) for target in targets], dtype=torch.float32
        ).reshape(-1, width),
    )


class SystemOneBatch(Batch):
    decision_loss_inputs: DecisionLossInputs


def system_one_collate(samples: list[Sample]) -> SystemOneBatch:
    """``cat_collate``, plus the micro-batch's decision-loss inputs."""
    batch = cat_collate(samples)
    inputs = build_decision_loss_inputs(batch["input_ids"][0], batch["sample_lens"].tolist(), batch["sample_data"])
    return SystemOneBatch(**batch, decision_loss_inputs=inputs)


def check_option_logits(logits: Any, questions: list[ReadQuestion], strategy_name: str) -> None:
    """Raise ``ValueError`` unless ``logits`` is a 1-D float tensor with one logit per option of ``questions``."""
    num_options = sum(len(question.options) for question in questions)
    if not isinstance(logits, Tensor) or not logits.is_floating_point() or logits.shape != (num_options,):
        got = f"{logits.dtype} of shape {tuple(logits.shape)}" if isinstance(logits, Tensor) else type(logits).__name__
        raise ValueError(
            f"{strategy_name}.option_logits must return a 1-D float tensor with the {num_options} option logits of "
            f"the micro-batch's {len(questions)} questions, got {got}"
        )


def checked_option_logits(
    strategy: SystemOneStrategy,
    model: nn.Module,
    hidden_states: Float[Tensor, "seq hidden"],
    input_ids: Int[Tensor, "seq"],
    questions: list[ReadQuestion],
) -> Float[Tensor, "options"]:
    """The strategy's ``option_logits``, checked."""
    logits = strategy.option_logits(model, hidden_states, input_ids, questions)
    check_option_logits(logits, questions, type(strategy).__name__)
    return logits


class DecisionLossOutput(TypedDict):
    """The decision loss of a micro-batch. A dict, so that FSDP's post-forward hook finds its tensors."""

    loss: Tensor
    """Summed over the labelled questions: soft cross-entropy plus ``brier_weight`` x Brier score."""
    ce: Tensor
    """Per labelled question: soft cross-entropy."""
    brier: Tensor
    """Per labelled question: Brier score."""
    accuracy: Tensor
    """Per labelled question: 1 when the top logit is at a top option of the target, else 0."""


DECISION_METRICS = ("ce", "brier", "accuracy")


def decision_loss(
    model: nn.Module,
    hidden_states: Float[Tensor, "1 seq hidden"],
    *,
    strategy: SystemOneStrategy,
    inputs: DecisionLossInputs,
    brier_weight: float,
) -> DecisionLossOutput:
    """Soft cross-entropy plus ``brier_weight`` x Brier score in float32, summed over the labelled questions. Used as
    the ``hidden_states_fn`` of ``forward``."""
    assert inputs.targets.dtype == torch.float32, f"targets must be float32, got {inputs.targets.dtype}"
    logits = checked_option_logits(strategy, model, hidden_states[0], inputs.input_ids, inputs.questions)
    targets = inputs.targets
    grid = logits.new_full((len(inputs.questions) * targets.shape[-1],), float("-inf"), dtype=torch.float32)
    logits = grid.scatter(0, inputs.option_slots, logits.float()).view(-1, targets.shape[-1])
    logits = logits.index_select(0, inputs.labelled)
    logp = logits.log_softmax(-1)
    # An option with target 0 adds nothing, even at a logit of -inf (padding included)
    ce = -torch.where(targets > 0, targets * logp, 0.0).sum(-1)
    brier = (logp.exp() - targets).square().sum(-1)
    # Any top option of the target counts, so a tie scores the same in any option order
    top = logits.argmax(-1, keepdim=True)
    accuracy = (targets.gather(-1, top).squeeze(-1) == targets.amax(-1)).float()
    return DecisionLossOutput(
        loss=ce.sum() + brier_weight * brier.sum(), ce=ce.detach(), brier=brier.detach(), accuracy=accuracy
    )
