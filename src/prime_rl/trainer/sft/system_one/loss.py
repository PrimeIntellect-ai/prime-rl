"""The System One contract for decision models: the strategy a model's class declares, the checked call to its
``option_logits``, and the loss such models share."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs
from prime_rl.trainer.sft.system_one.data import SystemOneStrategy


@runtime_checkable
class DecisionModel(Protocol):
    """What the class of a decision model (``[model.implementation]``) declares: its strategy, which renders records
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


def checked_option_logits(model: nn.Module, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
    """The logits of every question of the micro-batch: the ``option_logits`` of the model's strategy on the batch
    without the targets, checked to be one vector per question with one logit per option."""
    check_decision_model(model)
    records = batch["sample_data"]
    if records is None:
        raise ValueError('checked_option_logits needs System One data: set [data] type = "system_one"')
    unlabelled = [
        {
            **record,
            "questions": [{k: v for k, v in question.items() if k != "target"} for question in record["questions"]],
        }
        for record in records
    ]
    strategy_cls = type(model).system_one_strategy()
    logits = strategy_cls.option_logits(model, hidden_states, {**batch, "sample_data": unlabelled})
    questions = [(record["id"], question) for record in records for question in record["questions"]]
    if len(logits) != len(questions):
        raise ValueError(
            f"{strategy_cls.__name__}.option_logits returned {len(logits)} logit vectors for the {len(questions)} "
            f"questions of records {[record['id'] for record in records]}"
        )
    for (record_id, question), vector in zip(questions, logits):
        if vector.shape != (len(question["options"]),):
            raise ValueError(
                f"record {record_id!r}, question {question['id']!r}: {strategy_cls.__name__}.option_logits returned "
                f"logits of shape {tuple(vector.shape)} for {len(question['options'])} options"
            )
    return logits


def decision_loss(inputs: LossInputs, brier_weight: float = 0.0) -> LossOutputs:
    """The loss decision models share (``[loss] type = "decision"``):
    soft cross-entropy plus ``brier_weight`` x Brier score between each labelled question's distribution and its
    target, summed and counted per labelled question. Metrics: ``ce``, ``brier`` and ``accuracy`` (the top logit is
    at a top option of the target). The logits are ``checked_option_logits``, cast to float32.
    """
    logits = checked_option_logits(inputs.model, inputs.hidden_states, inputs.batch)
    questions = [question for record in inputs.batch["sample_data"] for question in record["questions"]]
    labelled = [
        (question["target"], vector) for question, vector in zip(questions, logits) if question["target"] is not None
    ]
    device = inputs.hidden_states.device

    # One host-to-device copy of all targets, so that the loop never waits for the device.
    host_targets = torch.tensor([p for target, _ in labelled for p in target], dtype=torch.float32)
    if device.type == "cuda":
        host_targets = host_targets.pin_memory()
    targets = host_targets.to(device, non_blocking=True).split([len(target) for target, _ in labelled])

    # Zero, but on every logit, so that backward runs on a micro-batch without a labelled question.
    loss = (torch.cat(logits).float() * 0.0).nan_to_num().sum()
    ce = brier = accuracy = torch.zeros((), device=device)
    for (_, vector), target in zip(labelled, targets):
        vector = vector.float()
        logp = vector.log_softmax(-1)
        # An option with target 0 adds nothing, even at a logit of -inf.
        question_ce = -torch.where(target > 0, target * logp, 0.0).sum()
        question_brier = (logp.exp() - target).square().sum()
        loss = loss + question_ce + brier_weight * question_brier
        ce, brier = ce + question_ce.detach(), brier + question_brier.detach()
        # A top option of the target, not the first one: a tied target is scored the same in any option order.
        accuracy = accuracy + (target.gather(0, vector.argmax(0, keepdim=True)) == target.max()).squeeze(0)
    return LossOutputs(loss=loss, count=len(labelled), metrics={"ce": ce, "brier": brier, "accuracy": accuracy})
