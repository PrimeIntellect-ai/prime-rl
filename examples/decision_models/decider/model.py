"""prime-rl's d3 class for a Decider release (``[model.implementation]``,
``import_path = "decision_models.decider.model.Decider"``)."""

from transformers import PretrainedConfig

from prime_rl.trainer.models.qwen3_5.modeling_d3 import D3ForSequenceClassification


class Decider(D3ForSequenceClassification):
    """d3's class, which a Decider release (perplexity-ai/pplx-decider-flash-v1.2) selects, with Decider's strategy,
    ``DeciderStrategy``. It reads ``decision_config.json`` format version 2 only: a d3 release (format 1) is served
    with d3's prompts, not Decider's. The release's layout and attention mode are d3's."""

    FORMAT_VERSIONS = (2,)

    @classmethod
    def system_one_strategy(cls):
        from decision_models.decider.strategy import DeciderStrategy

        return DeciderStrategy

    def export_config_files(self, config: PretrainedConfig) -> dict[str, dict]:
        """d3's, with temperature 1 in ``decision_config.json``: the logits trained leave the release's out."""
        files = super().export_config_files(config)
        files["decision_config.json"]["temperature"] = 1.0
        return files
