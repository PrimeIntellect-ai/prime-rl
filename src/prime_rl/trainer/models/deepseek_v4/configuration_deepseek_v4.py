from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_dict

CompressedLayerType = Literal["compressed_sparse_attention", "heavily_compressed_attention"]
DeepseekV4LayerType = Literal["sliding_attention", CompressedLayerType]

# `qk_rope_head_dim` (64) / `head_dim` (512).
DEFAULT_PARTIAL_ROTARY_FACTOR = 64 / 512


class DeepseekV4RopeParameters(BaseModel):
    """RoPE parameters keyed by rope type, independently of `layer_types`.

    Sliding-window layers rotate with `main`; the two compressed attention variants and their
    compressors share `compress`.
    """

    model_config = ConfigDict(extra="ignore")

    main: RopeParameters
    compress: RopeParameters


class DeepseekV4Config(PrimeModelConfig):
    """Configuration for DeepSeek-V4 (Flash / Pro) models.

    DeepSeek-V4 differs from V3 in three structural ways, all controlled from here:

    1. The residual stream is `hc_mult` parallel streams tied together by manifold-constrained
       hyper-connections (mHC), governed by `hc_mult`, `hc_sinkhorn_iters` and `hc_eps`.
    2. Attention is a per-layer mix of compressed variants selected by `layer_types`, with the
       per-type compression rate given by `compress_rates` and a Lightning Indexer sized by
       `index_n_heads` / `index_head_dim` / `index_topk`.
    3. The MLP schedule bootstraps the first `num_hash_layers` layers with a frozen hash router
       before switching to standard top-k routed MoE.

    Real checkpoints ship a legacy schema, which the `mode="before"` validator translates:

    - a flat per-layer `compress_ratios` list (0 / CSA rate / HCA rate) instead of `layer_types`;
    - `rope_theta`, `compress_rope_theta`, `partial_rotary_factor` (or `qk_rope_head_dim`) and a
      flat `rope_scaling` instead of the nested `rope_parameters`. The flat scaling applies to the
      `compress` rope only; `main` is always plain RoPE at `rope_theta`.
    """

    model_type: ClassVar[str] = "deepseek_v4"

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    rope_parameters: DeepseekV4RopeParameters
    max_position_embeddings: int = 1048576
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    layer_types: list[DeepseekV4LayerType] | None = None
    """Defaults to two heavily-compressed bootstrap layers, then alternating heavy/sparse."""
    compress_rates: dict[CompressedLayerType, int] = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    eos_token_id: int | list[int] | None = 1

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_schema(cls, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        compress_rates = data.get("compress_rates")
        if compress_rates is None:
            compress_rates = cls.model_fields["compress_rates"].default

        legacy_compress_ratios = data.pop("compress_ratios", None)
        if data.get("layer_types") is None and legacy_compress_ratios is not None:
            num_hidden_layers = data.get("num_hidden_layers", cls.model_fields["num_hidden_layers"].default)
            ratio_to_layer_type = {0: "sliding_attention", **{rate: lt for lt, rate in compress_rates.items()}}
            data["layer_types"] = [ratio_to_layer_type[ratio] for ratio in legacy_compress_ratios[:num_hidden_layers]]

        data["rope_parameters"] = cls._nest_rope_parameters(data)
        return data

    @classmethod
    def _nest_rope_parameters(cls, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Split RoPE parameters into the `main` and `compress` sets the rotary reads."""
        rope_parameters = data.get("rope_parameters")
        if rope_parameters is None:
            rope_parameters = data.get("rope_scaling")
        rope_parameters = rope_parameters or {}
        if isinstance(rope_parameters.get("main"), dict) and isinstance(rope_parameters.get("compress"), dict):
            return {
                "main": standardize_rope_dict(rope_parameters["main"], rope_theta=None),
                "compress": standardize_rope_dict(rope_parameters["compress"], rope_theta=None),
            }

        partial_rotary_factor = data.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            qk_rope_head_dim = data.get("qk_rope_head_dim")
            head_dim = data.get("head_dim", cls.model_fields["head_dim"].default)
            partial_rotary_factor = (
                qk_rope_head_dim / head_dim if qk_rope_head_dim is not None else DEFAULT_PARTIAL_ROTARY_FACTOR
            )

        scaling = {key: value for key, value in rope_parameters.items() if key not in ("main", "compress")}
        main = {
            "rope_type": "default",
            "rope_theta": data.get("rope_theta", 10000.0),
            "partial_rotary_factor": partial_rotary_factor,
        }
        compress = standardize_rope_dict(
            {**scaling, "rope_theta": data.get("compress_rope_theta", 160000.0)},
            rope_theta=None,
            partial_rotary_factor=partial_rotary_factor,
        )
        if compress["rope_type"] == "yarn":
            # The V4 reference does not scale cos/sin by YaRN's mscale; leaving the key unset would
            # derive `0.1 * log(factor) + 1 != 1.0`.
            compress.setdefault("attention_factor", 1.0)
        return {"main": main, "compress": compress}

    @model_validator(mode="after")
    def _resolve_and_validate_schedule(self) -> "DeepseekV4Config":
        if self.layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(self.num_hidden_layers - 2, 0))
            ]
            self.layer_types = ["heavily_compressed_attention"] * min(self.num_hidden_layers, 2) + interleave
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        for layer_type in set(self.layer_types) - {"sliding_attention"}:
            if layer_type not in self.compress_rates:
                raise ValueError(f"compress_rates is missing a rate for layer type {layer_type!r}.")
        if not 0 <= self.num_hash_layers <= self.num_hidden_layers:
            raise ValueError(
                f"num_hash_layers ({self.num_hash_layers}) must be between 0 and num_hidden_layers ({self.num_hidden_layers})."
            )
        return self


__all__ = ["DeepseekV4Config", "DeepseekV4LayerType", "DeepseekV4RopeParameters"]
