"""Decision 3.0 (d3): a Qwen3.5 backbone with an answer-code readout, the model vLLM 0.33 serves as
``D3ForSequenceClassification`` (``vllm/model_executor/models/d3.py``)."""

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig

from prime_rl.trainer.models.conversion_ops import Drop, PrefixRename
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

if TYPE_CHECKING:
    from prime_rl.trainer.sft.system_one.data import SystemOneStrategy

ATTENTION_MODES = ("causal", "noncausal_full_attention")


class D3ForSequenceClassification(Qwen3_5ForCausalLM):
    """prime-rl's Qwen3.5 with d3's readout, ``score``, which scores the answer codes at a prompt's last token.

    The loader builds this class for a ``Qwen3_5Model`` checkpoint with a ``decision_config.json`` beside its config,
    which it attaches as ``config.decision_config``, as vLLM does. A release stores the backbone under transformers'
    ``Qwen3_5Model`` names (``language_model.*``, ``visual.*``) and the readout as ``weight`` in
    ``readout.safetensors``. The class's HF names are those vLLM's d3 loads: ``language_model.*``, ``visual.*`` and
    ``score.weight``, with no LM head. The readout holds and computes in float32, as vLLM's does. prime-rl's Qwen3.5
    has an LM head, which d3 never calls: tied, it is the input embedding; untied, as in d3 and d3-flash, which ship
    none, it is a frozen copy of it, from ``init_missing_weights``.

    With ``attention_mode = "noncausal_full_attention"`` the full-attention layers attend in both directions within an
    attention segment; the Gated DeltaNet layers stay causal."""

    FORMAT_VERSIONS: tuple[int, ...] = (1,)
    """The ``decision_config.json`` format versions the class reads: vLLM 0.33's. A subclass may widen it."""

    fp32_modules = ("score",)
    export_weight_files = {"score.": "readout.safetensors"}

    def __init__(self, config) -> None:
        # vLLM's checks (vllm/model_executor/models/config.py, D3ForSequenceClassificationConfig)
        name = type(self).__name__
        decision = getattr(config, "decision_config", None)
        if not isinstance(decision, dict):
            raise ValueError(f"{name} needs a decision_config.json beside the checkpoint's config.json")
        if decision.get("format_version") not in self.FORMAT_VERSIONS:
            raise ValueError(
                f"unsupported decision_config.json format_version {decision.get('format_version')!r}; "
                f"{name} reads {list(self.FORMAT_VERSIONS)}"
            )
        if decision.get("pooling", "last") != "last":
            raise ValueError(f"unsupported decision_config.json pooling {decision['pooling']!r}")
        attention_mode = decision.get("attention_mode", "causal")
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(f"unsupported decision_config.json attention_mode {attention_mode!r}")
        codes, token_ids = decision.get("codes"), decision.get("token_ids")
        if not codes or not isinstance(token_ids, list) or len(codes) != len(token_ids):
            raise ValueError("decision_config.json must list one token id per answer code")
        temperature = decision.get("temperature", 1.0)
        if not isinstance(temperature, (int, float)) or not 0 < temperature < math.inf:
            raise ValueError(f"decision_config.json temperature must be a finite number above 0, got {temperature!r}")

        text_config = getattr(config, "text_config", config)
        text_config.is_causal = attention_mode == "causal"
        super().__init__(config)
        self.score = nn.Linear(text_config.hidden_size, len(codes), bias=False, dtype=torch.float32)
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.requires_grad_(False)

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        if getattr(config, "decision_config", {}).get("attention_mode", "causal") != "causal":
            return "its full-attention layers are non-causal (decision_config.json attention_mode)"
        return super().cp_blocker(config, cp)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name == "score.weight" or super().keep_in_fp32_for_weight_transfer(name)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.startswith("language_model.") or name == "weight" for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return not cls.is_hf_state_dict(state_dict) and any(
            name.startswith(("model.language_model.", "model.visual.", "score.")) for name in state_dict
        )

    @classmethod
    def conversion_chain(cls, config):
        return [
            PrefixRename("language_model.", "model.language_model."),
            PrefixRename("visual.", "model.visual."),
            Drop("lm_head.weight"),
            *super().conversion_chain(config),
        ]

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """The release's readout, ``weight`` in ``readout.safetensors``, becomes ``score.weight``: loading renames it,
        and the conversion to HF keeps vLLM's name, ``score.weight``."""
        if "weight" in state_dict:
            state_dict["score.weight"] = state_dict.pop("weight")
        return super().convert_to_prime(state_dict)

    def init_missing_weights(self, names: list[str]) -> dict[str, Tensor]:
        """An untied release's LM head: a copy of the loaded input embedding."""
        return {name: self.get_input_embeddings().weight for name in names if name == "lm_head.weight"}

    def export_config_files(self, config: PretrainedConfig) -> dict[str, dict]:
        """``decision_config.json``, taken off `config` with ``is_causal``, so the exported ``config.json`` is the
        release's."""
        vars(getattr(config, "text_config", config)).pop("is_causal", None)
        return {"decision_config.json": vars(config).pop("decision_config")}

    @classmethod
    def system_one_strategy(cls) -> type["SystemOneStrategy"]:
        """vLLM's read strategy for d3, without its softmax: ``D3Strategy``."""
        from prime_rl.trainer.sft.system_one.strategies.d3 import D3Strategy

        return D3Strategy


__all__ = ["D3ForSequenceClassification"]
