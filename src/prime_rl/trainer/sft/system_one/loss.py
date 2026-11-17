"""The System One contract for decision models: the strategy a model's class declares, the checked call to its
``option_logits``, and the loss such models share."""

from dataclasses import replace
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from prime_rl.trainer.sft.loss import LossInputs, LossOutputs
from prime_rl.trainer.sft.system_one.data import SystemOneStrategy


@runtime_checkable
class DecisionModel(Protocol):
    """What the class of a decision model (``[model.implementation]``) declares: its strategy, which renders requests
    and reads each question's logits with the class's modules. A class satisfies it by defining the classmethod,
    without inheriting from it."""

    @classmethod
    def system_one_strategy(cls) -> type[SystemOneStrategy]:
        """The class's strategy, imported inside the method so that the model module does not import the SFT data
        stack."""
        ...


def check_decision_model(model: nn.Module) -> None:
    """Raise unless the model's class declares its strategy."""
    if not isinstance(model, DecisionModel):
        raise TypeError(
            f"The decision loss needs a model whose class declares system_one_strategy, and {type(model).__name__} "
            "does not: name a class that declares it in [model.implementation]"
        )


def checked_option_logits(inputs: LossInputs) -> list[Tensor]:
    """The logits of one record's questions: the ``option_logits`` of the model's strategy on the record without its
    targets, checked to be one vector per question with one logit per option. It needs no labels, so evaluation code
    can call it on the ``LossInputs`` of one request rendered with ``render_sample`` and run alone."""
    check_decision_model(inputs.model)
    record = inputs.sample_data
    if record is None:
        raise ValueError('checked_option_logits needs System One data: set [data] type = "system_one"')
    if inputs.truncated:
        raise ValueError(f"record {record['id']!r} was cut by packing, so its read spans are incomplete")
    questions = record["questions"]
    unlabelled = {**record, "questions": [{k: v for k, v in q.items() if k != "target"} for q in questions]}
    strategy_cls = type(inputs.model).system_one_strategy()
    logits = strategy_cls.option_logits(replace(inputs, sample_data=unlabelled))
    if len(logits) != len(questions):
        raise ValueError(
            f"{strategy_cls.__name__}.option_logits returned {len(logits)} logit vectors for the {len(questions)} "
            f"questions of record {record['id']!r}"
        )
    for question, vector in zip(questions, logits):
        if vector.shape != (len(question["options"]),):
            raise ValueError(
                f"record {record['id']!r}, question {question['id']!r}: {strategy_cls.__name__}.option_logits "
                f"returned logits of shape {tuple(vector.shape)} for {len(question['options'])} options"
            )
    return logits


def decision_loss(inputs: LossInputs, brier_weight: float = 0.0) -> LossOutputs:
    """The loss decision models share (``[loss] type = "decision"``), on one record:
    soft cross-entropy plus ``brier_weight`` x Brier score between each labelled question's distribution and its
    target, summed and counted per labelled question. Metrics, one value per labelled question: ``ce``, ``brier`` and
    ``accuracy`` (the top logit is at a top option of the target). The logits are ``checked_option_logits``, cast to
    float32.
    """
    logits = checked_option_logits(inputs)
    labelled = [
        (question["target"], vector)
        for question, vector in zip(inputs.sample_data["questions"], logits)
        if question["target"] is not None
    ]
    device = inputs.hidden_states.device

    # One host-to-device copy of the record's targets, so that the loop never waits for the device.
    host_targets = torch.tensor([p for target, _ in labelled for p in target], dtype=torch.float32)
    if device.type == "cuda":
        host_targets = host_targets.pin_memory()
    targets = host_targets.to(device, non_blocking=True).split([len(target) for target, _ in labelled])

    # Zero, but on every logit, so that backward runs on a record without a labelled question.
    loss = (torch.cat(logits).float() * 0.0).nan_to_num().sum()
    ce, brier, accuracy = [], [], []
    for (_, vector), target in zip(labelled, targets):
        vector = vector.float()
        logp = vector.log_softmax(-1)
        # An option with target 0 adds nothing, even at a logit of -inf.
        question_ce = -torch.where(target > 0, target * logp, 0.0).sum()
        question_brier = (logp.exp() - target).square().sum()
        loss = loss + question_ce + brier_weight * question_brier
        ce.append(question_ce.detach())
        brier.append(question_brier.detach())
        # A top option of the target, not the first one: a tied target is scored the same in any option order.
        accuracy.append((target.gather(0, vector.argmax(0, keepdim=True)) == target.max()).squeeze(0).float())
    metrics = {
        name: torch.stack(values) if values else torch.zeros(0, device=device)
        for name, values in (("ce", ce), ("brier", brier), ("accuracy", accuracy))
    }
    return LossOutputs(loss=loss, count=len(labelled), metrics=metrics)
