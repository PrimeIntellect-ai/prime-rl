"""Qwen3.5 with Open-Jev's decision head (``[model.implementation]``,
``import_path = "decision_models.open_jev.model.Qwen3_5Decision"``)."""

import torch
from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


class Qwen3_5Decision(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``decision_head``, which scores one candidate prompt from its last token. The head's
    weights come from the checkpoint (``prepare_checkpoint.py``)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.decision_head = nn.Linear(self.lm_head.in_features, 1)

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: ``[0, score]`` for noul (options ``false``, ``true``), and
        the candidates' scores for choice and score. ``OpenJevEncoder`` reads one token per candidate prompt."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        reads = hidden_states[batch["loss_mask"]]
        scores = self.decision_head(reads.to(self.decision_head.weight.dtype)).squeeze(-1)
        logits = []
        for question, values in zip(questions, scores.split([len(question["reads"]) for question in questions])):
            logits.append(torch.cat([torch.zeros_like(values), values]) if question["type"] == "noul" else values)
        return logits
