"""Qwen3.5 with d3's answer-code readout (``[model.implementation]``,
``import_path = "decision_models.d3.model.Qwen3_5D3"``)."""

from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

ATTENTION_MODES = ("causal", "noncausal_full_attention")


class Qwen3_5D3(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``readout``, which scores d3's answer codes from a prompt's last token. The readout's
    weights and the ``d3`` entry of ``config.json`` (``readout_rows``, ``temperature``, ``attention_mode``) come from
    the checkpoint (``prepare_checkpoint.py``).

    With ``attention_mode = "noncausal_full_attention"`` the full-attention layers attend in both directions within a
    prompt (an attention segment); the Gated DeltaNet layers stay causal."""

    def __init__(self, config) -> None:
        super().__init__(config)
        d3 = config.d3
        if d3["attention_mode"] not in ATTENTION_MODES:
            raise ValueError(f"d3 attention_mode must be one of {ATTENTION_MODES}, got {d3['attention_mode']!r}")
        self.readout = nn.Linear(self.lm_head.in_features, d3["readout_rows"], bias=False)
        if d3["attention_mode"] == "noncausal_full_attention":
            for module in self.modules():
                if isinstance(module, Qwen3_5Attention):
                    module.is_causal = False

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: the readout's first rows, one per option, divided by the
        checkpoint's temperature. ``D3Encoder`` reads one token per question and lists the options under the answer
        codes in the order of the question's ``options``."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        reads = hidden_states[batch["loss_mask"]]
        logits = self.readout(reads.to(self.readout.weight.dtype)).float() / self.config.d3["temperature"]
        return [row[: len(question["options"])] for row, question in zip(logits, questions, strict=True)]
