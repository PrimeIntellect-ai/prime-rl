"""Qwen3.5 with Decider's answer-code readout (``[model.implementation]``,
``import_path = "decision_models.decider.model.Qwen3_5Decider"``)."""

from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

NUM_CODES = 255


class Qwen3_5Decider(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``readout``, which scores Decider's answer codes from a question's marker. The
    readout's weights come from the checkpoint (``prepare_checkpoint.py``).

    The full-attention layers attend in both directions within a record (an attention segment), so a marker sees
    the whole record; the Gated DeltaNet layers stay causal."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.readout = nn.Linear(self.lm_head.in_features, NUM_CODES, bias=False)
        for module in self.modules():
            if isinstance(module, Qwen3_5Attention):
                module.is_causal = False

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: the readout's first rows, one per option. ``DeciderEncoder``
        reads one token per question, its marker, and lists the options under the answer codes in the order of the
        question's ``options``."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        reads = hidden_states[batch["loss_mask"]]
        logits = self.readout(reads.to(self.readout.weight.dtype))
        return [row[: len(question["options"])] for row, question in zip(logits, questions, strict=True)]
