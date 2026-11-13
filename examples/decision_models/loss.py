"""The loss of every decision model in this folder: ``[loss] import_path = "decision_models.loss.decision_loss"``."""

import torch
import torch.nn.functional as F

from prime_rl.trainer.sft.loss import LossInputs, LossOutputs


def decision_loss(inputs: LossInputs, brier_weight: float = 0.0) -> LossOutputs:
    """Soft cross-entropy plus ``brier_weight`` x Brier score between each labelled question's distribution and its
    target, summed and counted per labelled question. Metrics: ``ce``, ``brier`` and ``accuracy`` (the top logit is
    a top option of the target).

    The logits come from the model class's ``decision_logits(hidden_states, batch)``: one 1-D tensor per question of
    the micro-batch (records in ``batch["sample_data"]`` order, questions in record order), with one logit per option
    in the order of the question's ``options``. It gets the batch without the targets.
    """
    records = inputs.batch["sample_data"]
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
        accuracy = accuracy + (target[vector.argmax()] == target.max())
        count += 1
    return LossOutputs(loss=loss, count=count, metrics={"ce": ce, "brier": brier, "accuracy": accuracy})
