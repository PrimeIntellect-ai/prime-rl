from typing import Any, ClassVar

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config, DeepseekV4RopeParameters


class DeepseekV41TextConfig(PrimeModelConfig):
    """Configuration for the DeepSeek-V4.1 language model.

    V4.1 keeps V4's shared-KV latent attention, mHC residual streams and sqrt-softplus MoE, and
    changes the long-range path and the residual wiring:

    - `compress_ratios[l]` is 0 (sliding window only), 1 or 2. Only the `kv_source_layer_ids`
      compress; every later layer of the same ratio reads the most recent source's entries.
    - Only the `index_source_layer_ids` run a Lightning Indexer; the layers in between reuse the
      most recent source's picks. Index keys come from the most recent KV source's entries.
    - Indexers after `candidate_source_layer_id` only score inside the `candidate_topk_blocks`
      blocks of `candidate_block_size` entries that the candidate source ranked highest.
    - Each sublayer's mHC `pre` gate collapses the streams for the *next* sublayer, and the
      final collapse uses the last FFN's gate: there is no `hc_head`.
    - Engram n-gram lookups are added into the residual stream before `engram_layer_ids`.

    `num_hash_layers` and `mlp_bias` exist only so the V4 MoE can be reused; V4.1 has neither.
    """

    model_type: ClassVar[str] = "deepseek_v41_text"

    name_or_path: str | None = None
    """Checkpoint the config was read from; the engram hash needs that checkpoint's tokenizer."""

    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    rope_parameters: DeepseekV4RopeParameters
    max_position_embeddings: int = 1048576
    sliding_window: int = 128
    compress_ratios: list[int]
    """One entry per layer, MTP layers included (only the first `num_hidden_layers` are used)."""
    kv_source_layer_ids: list[int] = [2, 8, 14, 20]
    index_source_layer_ids: list[int] = [2, 8, 14, 20, 24, 28, 32, 36]
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    """Negative disables candidate pre-filtering."""
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    num_experts_per_tok: int = 6
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: list[int] = [1, 14]
    engram_num_embeddings: list[int] = [384006168, 384016682]
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-20
    attention_dropout: float = 0.0
    eos_token_id: int | list[int] | None = 1
    num_hash_layers: int = 0
    mlp_bias: bool = False

    @model_validator(mode="before")
    @classmethod
    def _nest_rope_parameters(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Same legacy flat RoPE schema as V4: plain `main` RoPE for sliding layers, YaRN-scaled
        # `compress` RoPE for every compressed layer.
        data = dict(data)
        data["rope_parameters"] = DeepseekV4Config._nest_rope_parameters(data)
        return data

    @model_validator(mode="after")
    def _validate_schedule(self) -> "DeepseekV41TextConfig":
        n = self.num_hidden_layers
        if len(self.compress_ratios) < n:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {n} layers")
        for layer_idx in range(n):
            ratio = self.compress_ratios[layer_idx]
            if ratio not in (0, 1, 2):
                raise ValueError(f"layer {layer_idx} has compress ratio {ratio}; only 0, 1 and 2 are supported")
            if ratio and self.kv_source_of(layer_idx) is None:
                raise ValueError(f"compressed layer {layer_idx} has no KV source layer at or before it")
            if ratio and self.index_source_of(layer_idx) is None:
                raise ValueError(f"compressed layer {layer_idx} has no index source layer at or before it")
        for layer_idx in self.kv_source_layer_ids + self.index_source_layer_ids:
            if layer_idx < n and not self.compress_ratios[layer_idx]:
                raise ValueError(f"source layer {layer_idx} does not compress")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings must have the same length")
        return self

    def kv_source_of(self, layer_idx: int) -> int | None:
        """The layer whose compressed entries `layer_idx` reads: the most recent KV source of its ratio."""
        sources = [s for s in self.kv_source_layer_ids if s <= layer_idx]
        if not sources or self.compress_ratios[max(sources)] != self.compress_ratios[layer_idx]:
            return None
        return max(sources)

    def index_source_of(self, layer_idx: int) -> int | None:
        """The layer whose top-k picks `layer_idx` reads: the most recent index source."""
        sources = [s for s in self.index_source_layer_ids if s <= layer_idx]
        if not sources or self.compress_ratios[max(sources)] != self.compress_ratios[layer_idx]:
            return None
        return max(sources)

    def active_mm_params(self) -> int:
        """Parameters a token passes through in matmuls, for the trainer's MFU.

        The generic count in `perf.py` would read V4.1's low-rank query and grouped low-rank output
        projections as full-rank `hidden x heads x head_dim` matrices and miss the compressors,
        indexers, mHC projections and engram projections, so the architecture counts its own.
        """
        h, hd, nh = self.hidden_size, self.head_dim, self.num_attention_heads
        mix = (2 + self.hc_mult) * self.hc_mult
        params = self.vocab_size * h  # lm head
        for layer_idx in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer_idx]
            params += h * self.q_lora_rank + self.q_lora_rank * nh * hd + h * hd
            params += nh * hd * self.o_lora_rank + self.o_groups * self.o_lora_rank * h
            params += (self.num_experts_per_tok + self.n_shared_experts) * 3 * h * self.moe_intermediate_size
            params += self.n_routed_experts * h + 2 * mix * self.hc_mult * h
            if layer_idx in self.kv_source_layer_ids:
                params += h * hd * (2 if ratio > 1 else 1)
            if layer_idx in self.index_source_layer_ids:
                params += self.q_lora_rank * self.index_n_heads * self.index_head_dim + h * self.index_n_heads
                if layer_idx in self.kv_source_layer_ids:
                    params += hd * self.index_head_dim
        n_hash_cols = (self.engram_max_ngram_size - 1) * self.engram_n_heads
        params += len(self.active_engram_layer_ids) * n_hash_cols * self.engram_head_dim * h * (self.hc_mult + 1)
        return params

    @property
    def compress_rates(self) -> set[int]:
        return {r for r in self.compress_ratios[: self.num_hidden_layers] if r}

    @property
    def active_engram_layer_ids(self) -> list[int]:
        return [layer_idx for layer_idx in self.engram_layer_ids if layer_idx < self.num_hidden_layers]


class DeepseekV41Config(PrimeModelConfig):
    """Composite (`deepseek_v41`) wrapper; only the language model is trained."""

    model_type: ClassVar[str] = "deepseek_v41"

    text_config: DeepseekV41TextConfig

    @model_validator(mode="before")
    @classmethod
    def _forward_name_or_path(cls, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        if "name_or_path" in data and isinstance(data.get("text_config"), dict):
            data["text_config"] = {"name_or_path": data["name_or_path"], **data["text_config"]}
        return data

    @model_validator(mode="after")
    def _inherit_token_ids(self) -> "DeepseekV41Config":
        if "pad_token_id" not in self.model_fields_set:
            self.pad_token_id = self.text_config.pad_token_id
        if "eos_token_id" not in self.model_fields_set:
            self.eos_token_id = self.text_config.eos_token_id
        return self


__all__ = ["DeepseekV41Config", "DeepseekV41TextConfig"]
