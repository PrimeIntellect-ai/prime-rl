from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class GlmMoeDsaConfig(PrimeModelConfig):
    """GLM-5: DeepSeek-V3.2-style sparse MLA (DSA) attention with Mixture-of-Experts feed-forward layers.

    RoPE covers the ``qk_rope_head_dim`` slice of each head. The ``head_dim`` key some checkpoints
    (e.g. GLM-5.2) store is not read.
    """

    model_type: ClassVar[str] = "glm_moe_dsa"

    pad_token_id: int | None = 154820
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters
    attention_bias: bool = False

    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    qk_head_dim: int | None = None
    """Defaults to ``qk_nope_head_dim + qk_rope_head_dim``."""
    v_head_dim: int = 256

    # MoE
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    norm_topk_prob: bool = True

    # Sparse indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048

    # IndexCache: reuse top-k indices across layers. The trainer may set these after validation.
    use_index_cache: bool = False
    index_topk_freq: int = 1
    """Recompute top-k indices every ``index_topk_freq`` layers."""
    index_topk_pattern: str | None = None
    """Per-layer schedule overriding ``index_topk_freq``: ``"F"`` computes fresh indices, ``"S"`` reuses them."""
    indexer_types: list[Literal["full", "shared"]] | None = None
    """The checkpoint's IndexShare schedule (GLM-5.2): ``"shared"`` layers reuse indices and have no indexer."""

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "GlmMoeDsaConfig":
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        return self

    def skips_topk(self, layer_idx: int) -> bool:
        """Whether layer ``layer_idx`` reuses the cached top-k indices instead of running its indexer."""
        if not self.use_index_cache:
            return False
        if self.index_topk_pattern is not None:
            return layer_idx < len(self.index_topk_pattern) and self.index_topk_pattern[layer_idx] == "S"
        if self.indexer_types is not None:
            return layer_idx < len(self.indexer_types) and self.indexer_types[layer_idx] == "shared"
        return layer_idx % self.index_topk_freq != 0
