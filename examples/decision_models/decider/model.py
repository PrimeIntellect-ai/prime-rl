"""Qwen3.5 with Decider's answer-code readout (``[model.implementation]``,
``import_path = "decision_models.decider.model.Qwen3_5Decider"``)."""

from torch import Tensor, nn

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

NUM_CODES = 255
# A release names its backbone as a bare ``Qwen3_5Model`` does; prime-rl's names put ``model.`` before these.
RELEASE_PREFIXES = ("language_model.", "visual.")


class Qwen3_5Decider(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 plus ``readout``, which scores Decider's answer codes from a question's marker.

    ``model.name`` is a Decider release (perplexity-ai/pplx-decider-flash-v1.2), which holds the backbone under its
    own names and the readout apart, or a checkpoint under this class's names. The loader converts a release on the
    first load (``convert_to_prime``).

    The full-attention layers attend in both directions within a record (an attention segment), so a marker sees
    the whole record; the Gated DeltaNet layers stay causal."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.readout = nn.Linear(self.lm_head.in_features, NUM_CODES, bias=False)
        for module in self.modules():
            if isinstance(module, Qwen3_5Attention):
                module.is_causal = False

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Whether these are a release's names."""
        return any(name.startswith(RELEASE_PREFIXES) for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # The base class takes any Qwen3.5 layer names for prime-rl's, a release's included.
        return not cls.is_hf_state_dict(state_dict) and super().is_prime_state_dict(state_dict)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """A release's weights under this class's names: the backbone under ``model.``, and ``weight``, the readout
        in ``readout.safetensors``, as ``readout.weight``."""
        state_dict["readout.weight"] = state_dict.pop("weight")
        for name in [name for name in state_dict if name.startswith(RELEASE_PREFIXES)]:
            state_dict[f"model.{name}"] = state_dict.pop(name)
        return super().convert_to_prime(state_dict)

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: the readout's first rows, one per option. ``DeciderEncoder``
        reads one token per question, its marker, and lists the options under the answer codes in the order of the
        question's ``options``."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        reads = hidden_states[batch["loss_mask"]]
        logits = self.readout(reads.to(self.readout.weight.dtype))
        return [row[: len(question["options"])] for row, question in zip(logits, questions, strict=True)]
