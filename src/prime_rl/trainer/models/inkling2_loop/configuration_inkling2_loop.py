"""Config classes for the looped Inkling 2 checkpoints (``model_type: inkling2_loop``, e.g. Inkling 2 Mini).

Field names are the release ``config.json`` keys, so ``to_dict()`` stays a config the serving stack reads.
"""

from transformers.configuration_utils import PretrainedConfig

# Layers `core_start..core_end-1` run `num_passes` times with shared weights; `gate_rank` is the rank of the
# gates that mix the previous pass's messages into the next.
DEFAULT_LOOP_CONFIG = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


class Inkling2LoopTextConfig(PretrainedConfig):
    model_type = "inkling2_loop_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 2304,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 24,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 128,
        sliding_window_size: int = 1024,
        local_layer_ids: list[int] | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 8192,
        log_scaling_alpha: float = 0.1,
        rms_norm_eps: float = 1e-6,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        sconv_kernel_size: int = 4,
        n_layers_per_attn_res_block: int = 6,
        n_routed_experts: int = 128,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 1152,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        use_value_norm_no_gamma: bool = True,
        attn_res_normalize_values: bool = True,
        q_bias: bool = False,
        o_bias: bool = False,
        final_logit_softcapping: float = 30.0,
        loop_config: dict | None = None,
        model_max_length: int = 1048576,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 200006,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        # The checkpoint pads the unembedding to `vocab_size` rows; only the first `unpadded_vocab_size` are logits.
        # `None` is an unembedding without padding.
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        # Every sixth layer attends globally, the rest through the sliding window.
        self.local_layer_ids = (
            [layer_idx for layer_idx in range(num_hidden_layers) if (layer_idx + 1) % 6 != 0]
            if local_layer_ids is None
            else list(local_layer_ids)
        )
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.sconv_kernel_size = sconv_kernel_size
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.attn_res_normalize_values = attn_res_normalize_values
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.final_logit_softcapping = final_logit_softcapping
        self.loop_config = dict(DEFAULT_LOOP_CONFIG if loop_config is None else loop_config)
        self.model_max_length = model_max_length
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_logits(self) -> int:
        """Rows of the unembedding that are logits."""
        return self.vocab_size if self.unpadded_vocab_size is None else self.unpadded_vocab_size

    @property
    def layer_types(self) -> list[str]:
        local = set(self.local_layer_ids)
        return [
            SLIDING_ATTENTION if layer_idx in local else FULL_ATTENTION for layer_idx in range(self.num_hidden_layers)
        ]

    @layer_types.setter
    def layer_types(self, layer_types: list[str]) -> None:
        self.local_layer_ids = [
            layer_idx for layer_idx, layer_type in enumerate(layer_types) if layer_type == SLIDING_ATTENTION
        ]

    @property
    def loop_core(self) -> range:
        """The looped layers. A model truncated inside the core (``num_hidden_layers`` below ``core_end``) loops
        the whole blocks of the core that it has, and runs the layers after them once."""
        start, stop = self.loop_config["core_start"], self.loop_config["core_end"]
        if stop > self.num_hidden_layers:
            block = self.n_layers_per_attn_res_block
            stop = self.num_hidden_layers // block * block
        return range(min(start, stop), stop)

    @property
    def loop_num_passes(self) -> int:
        return self.loop_config["num_passes"]

    @property
    def loop_gate_rank(self) -> int:
        return self.loop_config["gate_rank"]

    def unsupported_variant(self) -> str | None:
        """Why PrimeRL cannot train this checkpoint variant, or ``None`` if it can."""
        released = dict(
            tie_word_embeddings=False,
            n_shared_experts=0,
            use_gate_bias=True,
            use_value_norm_no_gamma=True,
            attn_res_normalize_values=True,
            q_bias=False,
            o_bias=False,
        )
        for name, value in released.items():
            if getattr(self, name) != value:
                return f"{name}={getattr(self, name)!r} (only {value!r} is implemented)"
        if self.loop_config.get("is_varloop", False):
            return "loop_config.is_varloop=True (variable pass counts are not implemented)"
        if self.loop_num_passes < 1:
            return f"loop_config.num_passes={self.loop_num_passes} must be at least 1"
        core, block = self.loop_core, self.n_layers_per_attn_res_block
        if len(core) > 0 and self.loop_num_passes < 2:
            return f"loop_config.num_passes={self.loop_num_passes} (a loop core runs at least twice)"
        if len(core) > 0 and (core.start == 0 or core.start % block != 0 or core.stop % block != 0):
            return (
                f"the loop core {core.start}..{core.stop} must start after layer 0 and on attention-residual "
                f"block boundaries (blocks of {block} layers)"
            )
        if not 0 < self.num_logits <= self.vocab_size:
            return f"unpadded_vocab_size={self.unpadded_vocab_size} must be in 1..{self.vocab_size}"
        return None


class Inkling2LoopConfig(PretrainedConfig):
    """Composite checkpoint config. Only the text model is trained; the vision and draft configs are kept as-is."""

    model_type = "inkling2_loop"
    sub_configs = {"text_config": Inkling2LoopTextConfig}

    def __init__(
        self,
        text_config: Inkling2LoopTextConfig | dict | None = None,
        vision_config: dict | None = None,
        mtp_config: dict | None = None,
        image_token_id: int = 200054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Inkling2LoopTextConfig(**text_config)
        self.text_config = text_config or Inkling2LoopTextConfig()
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        self.image_token_id = image_token_id
        # The release's `quantization_config` describes its packed experts. PrimeRL dequantizes them on load.
        kwargs.pop("quantization_config", None)
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("bos_token_id", self.text_config.bos_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["FULL_ATTENTION", "SLIDING_ATTENTION", "Inkling2LoopConfig", "Inkling2LoopTextConfig"]
