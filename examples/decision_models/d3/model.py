"""Qwen3.5 with d3's answer-code readout (``[model.implementation]``,
``import_path = "decision_models.d3.model.Qwen3_5D3"``)."""

from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

ATTENTION_MODES = ("causal", "noncausal_full_attention")


class Qwen3_5D3(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``readout``, which scores d3's answer codes from a prompt's last token. The ``d3``
    entry of the checkpoint's ``config.json`` (``readout_rows``, ``temperature``, ``attention_mode``) is written by
    ``write_config.py``.

    A d3 release stores the backbone under the names of transformers' ``Qwen3_5Model`` (``language_model.*``,
    ``visual.*``) and the readout apart, as ``weight`` in ``readout.safetensors``. The loader converts it once, on
    the first load (``convert_to_prime``); a checkpoint under this class's names loads as it is.

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

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Whether these are a d3 release's names: the backbone without the ``model.`` prefix."""
        return any(name.startswith("language_model.") for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return not cls.is_hf_state_dict(state_dict) and super().is_prime_state_dict(state_dict)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """A release's weights under this class's names: the backbone under ``model.``, the readout as
        ``readout.weight``."""
        state_dict["readout.weight"] = state_dict.pop("weight")
        for name in [name for name in state_dict if name.startswith(("language_model.", "visual."))]:
            state_dict[f"model.{name}"] = state_dict.pop(name)
        return super().convert_to_prime(state_dict)

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: the readout's first rows, one per option, divided by the
        checkpoint's temperature. ``D3Encoder`` reads one token per question and lists the options under the answer
        codes in the order of the question's ``options``."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        reads = hidden_states[batch["loss_mask"]]
        logits = self.readout(reads.to(self.readout.weight.dtype)).float() / self.config.d3["temperature"]
        return [row[: len(question["options"])] for row, question in zip(logits, questions, strict=True)]
