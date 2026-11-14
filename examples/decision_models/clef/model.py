"""Qwen3.5 with Clef's joint schema head (``[model.implementation]``,
``import_path = "decision_models.clef.model.Qwen3_5Clef"``)."""

import torch
from decision_models.clef.joint_schema_model import QUESTION_TYPES, EncodedQuestion, EncodedRecord, JointSchemaHead
from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

SCALARS = ("prior_logit_scale", "joint_logit_scale", "residual_gate")


class Qwen3_5Clef(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus two modules, whose weights come from the checkpoint (``prepare_checkpoint.py``):

    ``joint_head``         Clef's ``JointSchemaHead``, sized by the ``joint_head`` entry of the checkpoint's config
    ``lexical_embedding``  a frozen copy of the LM head's rows, for the head's lexical prior. The prior indexes those
                           rows, and docs/training.md ("Custom Loss") says to call the LM head, not to read its
                           weight. The copy is in the root FSDP unit, which the forward pass gathers.

    FSDP refuses 0-dimensional parameters, so the head's three scalars hold one element each. The head only multiplies
    them into per-option vectors, so its outputs are unchanged.

    The LM head is frozen too. The loss never calls it, so it would get no gradient, and a trainable parameter without
    optimizer state breaks checkpoint resume. Clef's backbones have an untied LM head (``prepare_checkpoint.py``
    requires one), so this freezes nothing else, and the lexical copy stays equal to it.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.joint_head.get("dropout"):
            # prime-rl never switches the model to eval mode, so validation would run with the dropout on.
            raise ValueError(f"Qwen3_5Clef needs a joint head without dropout, got {config.joint_head['dropout']}")
        self.joint_head = JointSchemaHead(**config.joint_head)
        for name in SCALARS:
            setattr(self.joint_head, name, nn.Parameter(getattr(self.joint_head, name).detach().reshape(1)))
        self.lexical_embedding = nn.Embedding(self.lm_head.out_features, self.lm_head.in_features)
        self.lexical_embedding.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order. The head runs once per record, on the record's own slice of
        the packed row: it attends over the whole record and decides the record's questions together.
        ``ClefEncoder`` reads a question's instruction span, then one span per option."""
        hidden_states = hidden_states.to(self.joint_head.hidden_norm.weight.dtype)
        input_ids = batch["input_ids"]
        logits, start = [], 0
        for record in batch["sample_data"]:
            end = start + record["num_tokens"]
            questions = tuple(
                EncodedQuestion(
                    question_id=question["id"],
                    question_type=QUESTION_TYPES[question["type"]],
                    question_span=tuple(question["reads"][0]),
                    option_spans=tuple(tuple(span) for span in question["reads"][1:]),
                    option_ids=tuple(question["options"]),
                )
                for question in record["questions"]
            )
            (record_logits,) = self.joint_head(
                hidden_states[:, start:end],
                input_ids[:, start:end],
                torch.ones_like(input_ids[:, start:end]),
                [EncodedRecord(input_ids=(), questions=questions, record_id=str(record["id"]))],
                self.lexical_embedding.weight,
            )
            logits += record_logits
            start = end
        return logits
