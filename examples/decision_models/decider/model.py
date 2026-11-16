"""prime-rl's d3 class for a Decider release (``[model.implementation]``,
``import_path = "decision_models.decider.model.Decider"``)."""

from torch import Tensor

from prime_rl.trainer.data import Batch
from prime_rl.trainer.models.qwen3_5.modeling_d3 import D3ForSequenceClassification


class Decider(D3ForSequenceClassification):
    """d3's class, which a Decider release (perplexity-ai/pplx-decider-flash-v1.2) selects, reading its
    ``decision_config.json`` format version 2 only: a d3 release (format 1) is served with d3's prompts, not
    Decider's. The release's layout and attention mode are d3's."""

    FORMAT_VERSIONS = (2,)

    def export_config_files(self) -> dict[str, dict]:
        """d3's, with temperature 1 in ``decision_config.json``: the logits trained leave the release's out."""
        files = super().export_config_files()
        files["decision_config.json"]["temperature"] = 1.0
        return files

    def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]:
        """One logit vector per question, in pack order: the readout's first rows, one per option, at the question's
        marker, which ``DeciderEncoder`` reads. Unlike d3's, they are not divided by the temperature."""
        questions = [question for record in batch["sample_data"] for question in record["questions"]]
        logits = self.score(hidden_states[batch["loss_mask"]].float())
        return [row[: len(question["options"])] for row, question in zip(logits, questions, strict=True)]
