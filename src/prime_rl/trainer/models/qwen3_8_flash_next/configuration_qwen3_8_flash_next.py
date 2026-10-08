from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters

LayerType = Literal["linear_attention", "full_attention"]


class Qwen3_8FlashNextRopeParameters(RopeParameters):
    rope_type: Literal["default"] = "default"
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)


class Qwen3_8FlashNextTextConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "qwen4_exp_text"

    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_parameters: Qwen3_8FlashNextRopeParameters
    layer_types: list[LayerType] | None = None
    """Defaults to full attention on every ``full_attention_interval``-th layer, linear attention elsewhere."""
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: list[int] = [2]
    """1-based indices of the layers that apply position learning enhancement."""
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    num_experts_per_tok: int = 10
    num_experts: int = 512
    load_balance_coeff: float | None = None
    eos_token_id: int = 248044

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        # RoPE is read from `rope_parameters` only; a top-level `partial_rotary_factor` is the fallback.
        rope = dict(data.get("rope_parameters") or {})
        if "partial_rotary_factor" in data:
            rope.setdefault("partial_rotary_factor", data["partial_rotary_factor"])
        return {**data, "rope_parameters": rope}

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Qwen3_8FlashNextTextConfig":
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (layer_index + 1) % self.full_attention_interval else "full_attention"
                for layer_index in range(self.num_hidden_layers)
            ]
        return self


class Qwen3_8FlashNextConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "qwen4_exp"

    text_config: Qwen3_8FlashNextTextConfig = Field(default_factory=Qwen3_8FlashNextTextConfig)

    @model_validator(mode="after")
    def _inherit_token_ids(self) -> "Qwen3_8FlashNextConfig":
        if "pad_token_id" not in self.model_fields_set:
            self.pad_token_id = self.text_config.pad_token_id
        if "eos_token_id" not in self.model_fields_set:
            self.eos_token_id = self.text_config.eos_token_id
        return self


__all__ = ["Qwen3_8FlashNextConfig", "Qwen3_8FlashNextRopeParameters", "Qwen3_8FlashNextTextConfig"]
