"""Qwen3.5 with Open-Jev's decision head (``[model.implementation]``,
``import_path = "decision_models.open_jev.model.Qwen3_5Decision"``)."""

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} must be a single token for the Yes/No initialisation, got {ids}")
    return ids[0]


class Qwen3_5Decision(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``decision_head``, which scores one candidate prompt from its last token, as
    ``OpenJevStrategy`` reads it. The base model has no head: ``init_missing_weights`` gives the weights it starts
    with."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.decision_head = nn.Linear(self.lm_head.in_features, 1)

    @classmethod
    def system_one_strategy(cls):
        from decision_models.open_jev.strategy import OpenJevStrategy

        return OpenJevStrategy

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
