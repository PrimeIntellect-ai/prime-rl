from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PretrainedConfig

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

# DeepSeek-V4.1-Flash's attention schedule: two sliding-window layers, 18 layers pooling pairs of tokens,
# 20 layers keeping one entry per token. The last three entries are the DSpark draft layers, which are
# not trained.
DEFAULT_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3


@strict
class DeepseekV41TextConfig(PretrainedConfig):
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

    RoPE follows V4: the `main` rope for sliding-window layers, the YaRN-scaled `compress` rope for
    every compressed layer. `num_hash_layers` and `mlp_bias` are fixed attributes that only exist so
    V4's MoE can be reused; V4.1 has neither.

    Args:
        compress_ratios: One entry per layer, DSpark draft layers included; only the first
            `num_hidden_layers` are read.
        kv_source_layer_ids: Layers that compress the token stream for themselves and later layers.
        index_source_layer_ids: Layers that run a Lightning Indexer for themselves and later layers.
        candidate_source_layer_id: Layer whose indexer also ranks blocks of entries for every later
            indexer. Negative disables candidate pre-filtering.
        candidate_topk_blocks: Blocks each query keeps from the candidate source's ranking.
        candidate_block_size: Entries per candidate block.
        engram_layer_ids: Layers whose input gets an engram lookup added.
        engram_num_embeddings: Rows of each engram layer's table.
        engram_max_ngram_size: Longest n-gram hashed; every size from 2 up to it is.
        engram_vocab_size: Lower bound of the prime bucket sizes.
        engram_n_heads: Hash heads per n-gram size.
        engram_head_dim: Width of a table row.
        engram_pad_token_id: Token whose compressed id fills look-back before a document start.
        engram_compressed_vocab_size: Size of the normalized token space the n-grams hash over.

    The remaining arguments are V4's (see `DeepseekV4Config`).
    """

    model_type = "deepseek_v41_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    # RoPE is keyed by rope type, exactly as in V4.
    _rope_type_labels = DeepseekV4Config._rope_type_labels
    default_partial_rotary_factor = DeepseekV4Config.default_partial_rotary_factor
    _nest_rope_parameters = DeepseekV4Config._nest_rope_parameters
    convert_rope_params_to_dict = DeepseekV4Config.convert_rope_params_to_dict
    validate_rope = DeepseekV4Config.validate_rope

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        partial_rotary_factor: float | None = None,
        qk_rope_head_dim: int | None = None,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 1048576,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        kv_source_layer_ids: list[int] | None = None,
        index_source_layer_ids: list[int] | None = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        engram_layer_ids: list[int] | None = None,
        engram_num_embeddings: list[int] | None = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-20,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        kv_precision: str = "bf16",
        dsa_backend: str = "tilelang",
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        **kwargs,
    ):
        raw_rope_parameters = rope_parameters if rope_parameters is not None else rope_scaling

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Attention
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.kv_precision = kv_precision
        self.dsa_backend = dsa_backend
        self.partial_rotary_factor = (
            partial_rotary_factor
            if partial_rotary_factor is not None
            else (qk_rope_head_dim / head_dim if qk_rope_head_dim is not None else self.default_partial_rotary_factor)
        )
        self.qk_rope_head_dim = int(head_dim * self.partial_rotary_factor)

        # Shared compressed KV, Lightning Indexers and candidate blocks
        self.compress_ratios = list(compress_ratios if compress_ratios is not None else DEFAULT_COMPRESS_RATIOS)
        self.kv_source_layer_ids = list(kv_source_layer_ids if kv_source_layer_ids is not None else [2, 8, 14, 20])
        self.index_source_layer_ids = list(
            index_source_layer_ids if index_source_layer_ids is not None else [2, 8, 14, 20, 24, 28, 32, 36]
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size

        # MoE
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.num_hash_layers = 0
        self.mlp_bias = False

        # Manifold-constrained hyper-connections
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # Engram
        self.engram_layer_ids = list(engram_layer_ids if engram_layer_ids is not None else [1, 14])
        self.engram_num_embeddings = list(
            engram_num_embeddings if engram_num_embeddings is not None else [384006168, 384016682]
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.rope_parameters = self._nest_rope_parameters(raw_rope_parameters)

    def validate_schedule(self) -> None:
        """Every compressed layer must have a KV and an index source of its own ratio at or before it."""
        n = self.num_hidden_layers
        if len(self.compress_ratios) < n:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {n} layers")
        for layer_idx in range(n):
            ratio = self.compress_ratios[layer_idx]
            if ratio not in (0, 1, 2):
                raise ValueError(f"layer {layer_idx} has compress ratio {ratio}; only 0, 1 and 2 are supported")
            if ratio and self.kv_source_of(layer_idx) is None:
                raise ValueError(f"compressed layer {layer_idx} has no KV source layer of its ratio at or before it")
            if ratio and self.index_source_of(layer_idx) is None:
                raise ValueError(f"compressed layer {layer_idx} has no index source layer of its ratio at or before it")
        for layer_idx in self.kv_source_layer_ids + self.index_source_layer_ids:
            if layer_idx < n and not self.compress_ratios[layer_idx]:
                raise ValueError(f"source layer {layer_idx} does not compress")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings must have the same length")

    def kv_source_of(self, layer_idx: int) -> int | None:
        """The layer whose compressed entries `layer_idx` reads: the most recent KV source of its ratio."""
        return self._latest_source(self.kv_source_layer_ids, layer_idx)

    def index_source_of(self, layer_idx: int) -> int | None:
        """The layer whose top-k picks `layer_idx` reads: the most recent index source of its ratio."""
        return self._latest_source(self.index_source_layer_ids, layer_idx)

    def _latest_source(self, source_ids: list[int], layer_idx: int) -> int | None:
        sources = [source for source in source_ids if source <= layer_idx]
        if not sources or self.compress_ratios[max(sources)] != self.compress_ratios[layer_idx]:
            return None
        return max(sources)

    @property
    def compress_rates(self) -> set[int]:
        """The compress ratios in use, each with its own entry layout."""
        return {ratio for ratio in self.compress_ratios[: self.num_hidden_layers] if ratio}

    @property
    def active_engram_layer_ids(self) -> list[int]:
        return [layer_idx for layer_idx in self.engram_layer_ids if layer_idx < self.num_hidden_layers]

    def active_mm_params(self) -> int:
        """Parameters a token passes through in matmuls, for the trainer's MFU.

        The generic count in `perf.py` would read V4.1's low-rank query and grouped low-rank output
        projections as full-rank `hidden x heads x head_dim` matrices and miss the compressors,
        indexers, mHC projections and engram projections, so the architecture counts its own.
        """
        h, hd, nh = self.hidden_size, self.head_dim, self.num_attention_heads
        mhc_params = (2 + self.hc_mult) * self.hc_mult**2 * h
        params = self.vocab_size * h  # lm head
        for layer_idx in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer_idx]
            params += h * self.q_lora_rank + self.q_lora_rank * nh * hd + h * hd
            params += nh * hd * self.o_lora_rank + self.o_groups * self.o_lora_rank * h
            params += (self.num_experts_per_tok + self.n_shared_experts) * 3 * h * self.moe_intermediate_size
            params += self.n_routed_experts * h + 2 * mhc_params
            if layer_idx in self.kv_source_layer_ids:
                params += h * hd * (2 if ratio > 1 else 1)
            if layer_idx in self.index_source_layer_ids:
                params += self.q_lora_rank * self.index_n_heads * self.index_head_dim + h * self.index_n_heads
                if layer_idx in self.kv_source_layer_ids:
                    params += hd * self.index_head_dim
        n_hash_cols = (self.engram_max_ngram_size - 1) * self.engram_n_heads
        params += len(self.active_engram_layer_ids) * n_hash_cols * self.engram_head_dim * h * (self.hc_mult + 1)
        return params


@strict
class DeepseekV41Config(PretrainedConfig):
    """The published composite `deepseek_v41` config. Only its `text_config` is trained: the vision
    tower and the DSpark draft head the checkpoint also carries are dropped."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(
        self,
        text_config: DeepseekV41TextConfig | dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        self.text_config = text_config if text_config is not None else DeepseekV41TextConfig()
        kwargs.pop("vision_config", None)
        kwargs.pop("image_token_id", None)
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("bos_token_id", self.text_config.bos_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["DeepseekV41Config", "DeepseekV41TextConfig"]
