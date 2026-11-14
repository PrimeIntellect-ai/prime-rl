"""Qwen3.5 with Open-Jev's decision head (``[model.implementation]``,
``import_path = "decision_models.open_jev.model.Qwen3_5Decision"``)."""

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} must be a single token for the Yes/No initialisation, got {ids}")
    return ids[0]


class Qwen3_5Decision(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``decision_head``, which scores one candidate prompt from its last token. The base
    model has no head: ``init_missing_weights`` gives the weights it starts with."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.decision_head = nn.Linear(self.lm_head.in_features, 1)

    def init_missing_weights(self, names: list[str]) -> dict[str, Tensor]:
        """Weight = lm_head[Yes] - lm_head[No] and bias = 0: the untrained head scores a candidate as
        logit(Yes) - logit(No), a zero-shot judge, which is how Open-Jev initialises it."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
        yes, no = token_id(tokenizer, "Yes"), token_id(tokenizer, "No")
        rows = self.lm_head.weight  # with tied embeddings, the input embedding
        if isinstance(rows, DTensor):
            rows = rows.full_tensor()
        weight = rows[yes : yes + 1].float() - rows[no : no + 1].float()
        return {"decision_head.weight": weight, "decision_head.bias": torch.zeros(1)}

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
