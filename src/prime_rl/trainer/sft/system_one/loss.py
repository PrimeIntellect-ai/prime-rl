"""The System One contract for decision models: the readout a model's class defines, and the loss they share."""

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor

from prime_rl.trainer.data import Batch
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs


@runtime_checkable
class DecisionModel(Protocol):
    """What the class of a decision model (``[model.implementation]``) defines: its readout, from hidden states to
    the logits of each question. A class satisfies it by defining the method, without inheriting from it."""

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """The logits of every question of one packed row: one 1-D tensor per question, in pack order (records in
        ``batch["sample_data"]`` order, questions in record order), with one logit per option in the order of the
        question's ``options``. Their softmax is the question's distribution.

        ``hidden_states`` are the backbone's final (post-norm) hidden states, ``(1, tokens, hidden)``. ``batch`` is the
        micro-batch on the device, and its questions carry no ``target``. A record's logits must not depend on the
        other records of the row.

        Called inside the model's forward pass: call the model's modules rather than reading their weights, on inputs
        in their weights' dtype. The logits may have any float dtype."""
        ...


def decision_loss(inputs: LossInputs, brier_weight: float = 0.0) -> LossOutputs:
    """The loss decision models share (``[loss] import_path = "prime_rl.trainer.sft.system_one.loss.decision_loss"``):
    soft cross-entropy plus ``brier_weight`` x Brier score between each labelled question's distribution and its
    target, summed and counted per labelled question. Metrics: ``ce``, ``brier`` and ``accuracy`` (the top logit is
    at a top option of the target).

    The logits are the model's ``decision_logits`` on the batch without the targets, checked to be one vector per
    question with one logit per option, and cast to float32.
    """
    if not isinstance(inputs.model, DecisionModel):
        raise TypeError(
            f"decision_loss needs a model whose class defines decision_logits(hidden_states, batch), and "
            f"{type(inputs.model).__name__} does not: name a class that defines it in [model.implementation]"
        )
    records = inputs.batch["sample_data"]
    if records is None:
        raise ValueError('decision_loss needs System One data: set [data] type = "system_one"')
    unlabelled = [
        {
            **record,
            "questions": [{k: v for k, v in question.items() if k != "target"} for question in record["questions"]],
        }
        for record in records
    ]
    logits = inputs.model.decision_logits(inputs.hidden_states, {**inputs.batch, "sample_data": unlabelled})
    questions = [(record["id"], question) for record in records for question in record["questions"]]
    if len(logits) != len(questions):
        raise ValueError(
            f"decision_logits returned {len(logits)} logit vectors for the {len(questions)} questions of records "
            f"{[record['id'] for record in records]}"
        )

    loss = ce = brier = accuracy = torch.zeros((), device=inputs.hidden_states.device)
    count = 0
    for (record_id, question), vector in zip(questions, logits):
        if vector.shape != (len(question["options"]),):
            raise ValueError(
                f"record {record_id!r}, question {question['id']!r}: decision_logits returned logits of shape "
                f"{tuple(vector.shape)} for {len(question['options'])} options"
            )
        if question["target"] is None:
            continue
        vector = vector.float()
        target = torch.tensor(question["target"], device=vector.device)
        question_ce = F.cross_entropy(vector, target)
        question_brier = (vector.softmax(-1) - target).square().sum()
        loss = loss + question_ce + brier_weight * question_brier
        ce, brier = ce + question_ce.detach(), brier + question_brier.detach()
        # A top option of the target, not the first one: a tied target is scored the same in any option order.
        accuracy = accuracy + (target[vector.argmax()] == target.max())
        count += 1
    return LossOutputs(loss=loss, count=count, metrics={"ce": ce, "brier": brier, "accuracy": accuracy})
