"""The System One contract for decision models: the readout a model's class defines, the checked call to it, and the
loss such models share."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs


@runtime_checkable
class DecisionModel(Protocol):
    """What the class of a decision model (``[model.implementation]``) defines: its readout, from hidden states to
    the logits of each question. A class satisfies it by defining the method, without inheriting from it."""

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """The logits of every question of one packed row: one 1-D tensor per question, in pack order (records in
        ``batch["sample_data"]`` order, questions in record order), with one logit per option in the order of the
        question's ``options``. They are the logits the model trains on: a checkpoint's own temperature, which its
        server applies, is part of these logits; a calibration fitted after training is not.

        ``hidden_states`` are the backbone's final (post-norm) hidden states, ``(1, tokens, hidden)``. ``batch`` is the
        micro-batch on the device. A readout may rely on its ``input_ids``, ``loss_mask`` and ``sample_data``, whose
        records hold ``id`` and ``num_tokens`` and whose questions hold ``id``, ``type``, ``options`` and ``reads``,
        and never a ``target``. A record's logits must not depend on the other records of the row.

        Called inside the model's forward pass: call the model's modules rather than reading their weights, on inputs
        in their weights' dtype. The logits may have any float dtype."""
        ...


def question_logits(model: nn.Module, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
    """The model's logits for every question of the micro-batch: its ``decision_logits`` on the batch without the
    targets, checked to be one vector per question with one logit per option."""
    if not isinstance(model, DecisionModel):
        raise TypeError(
            f"question_logits needs a model whose class defines decision_logits(hidden_states, batch), and "
            f"{type(model).__name__} does not: name a class that defines it in [model.implementation]"
        )
    records = batch["sample_data"]
    if records is None:
        raise ValueError('question_logits needs System One data: set [data] type = "system_one"')
    unlabelled = [
        {
            **record,
            "questions": [{k: v for k, v in question.items() if k != "target"} for question in record["questions"]],
        }
        for record in records
    ]
    logits = model.decision_logits(hidden_states, {**batch, "sample_data": unlabelled})
    questions = [(record["id"], question) for record in records for question in record["questions"]]
    if len(logits) != len(questions):
        raise ValueError(
            f"decision_logits returned {len(logits)} logit vectors for the {len(questions)} questions of records "
            f"{[record['id'] for record in records]}"
        )
    for (record_id, question), vector in zip(questions, logits):
        if vector.shape != (len(question["options"]),):
            raise ValueError(
                f"record {record_id!r}, question {question['id']!r}: decision_logits returned logits of shape "
                f"{tuple(vector.shape)} for {len(question['options'])} options"
            )
    return logits


def decision_loss(inputs: LossInputs, brier_weight: float = 0.0) -> LossOutputs:
    """The loss decision models share (``[loss] type = "decision"``):
    soft cross-entropy plus ``brier_weight`` x Brier score between each labelled question's distribution and its
    target, summed and counted per labelled question. Metrics: ``ce``, ``brier`` and ``accuracy`` (the top logit is
    at a top option of the target). The logits are the model's ``question_logits``, cast to float32.
    """
    logits = question_logits(inputs.model, inputs.hidden_states, inputs.batch)
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
