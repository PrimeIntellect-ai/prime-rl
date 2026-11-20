"""Configuration for Thinking Machines' Inkling 2 (text tower)."""

from transformers.configuration_utils import PretrainedConfig


class Inkling2TextConfig(PretrainedConfig):
    """Text configuration for Inkling 2.

    The release checkpoint is a single multimodal ``Inkling2Config`` whose
    ``text_config`` holds these fields; this class is the flat text view used by
    the trainer (vision tower and DSpark/MTP head are not trained).
    """

    model_type = "inkling2_text"

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        d_rel: int = 16,
        rel_extent: int = 1024,
        sliding_window_size: int = 1024,
        swa_num_attention_heads: int = 48,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 128,
        local_layer_ids: list[int] | None = None,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        sconv_kernel_size: int = 4,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 2304,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        log_scaling_n_floor: int = 128000,
        log_scaling_alpha: float = 0.1,
        final_logit_softcapping: float = 30.0,
        rms_norm_eps: float = 1e-6,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        q_bias: bool = False,
        o_bias: bool = False,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 200006,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = vocab_size if unpadded_vocab_size is None else unpadded_vocab_size
        if not 0 < self.unpadded_vocab_size <= vocab_size:
            raise ValueError("unpadded_vocab_size must be between 1 and vocab_size")
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.sliding_window_size = sliding_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.local_layer_ids = (
            list(local_layer_ids)
            if local_layer_ids is not None
            else [i for i in range(num_hidden_layers) if (i + 1) % n_layers_per_attn_res_block != 0]
        )
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.sconv_kernel_size = sconv_kernel_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.final_logit_softcapping = final_logit_softcapping
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.initializer_range = initializer_range
        self.layer_types = [
            "hybrid_sliding" if i in set(self.local_layer_ids) else "hybrid" for i in range(num_hidden_layers)
        ]
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class Inkling2Config(Inkling2TextConfig):
    """Flat view of the release's multimodal config (``model_type="inkling2"``).

    ``text_config`` from the checkpoint is merged into this class's fields; the
    vision tower and ``mtp_config`` are ignored (text-only training).
    """

    model_type = "inkling2"

    def __init__(self, text_config: dict | None = None, vision_config: dict | None = None, **kwargs):
        merged = dict(kwargs)
        if isinstance(text_config, dict):
            merged = {**text_config, **merged}
        merged.pop("torch_dtype", None)
        merged.pop("model_max_length", None)
        if "intermediate_size" in merged:
            merged.setdefault("moe_intermediate_size", merged.pop("intermediate_size"))
        # The release config carries these vision/MTP-only keys at the top level.
        for key in ("mtp_config", "vision_config", "architectures", "quantization_config", "transformers_version"):
            merged.pop(key, None)
        super().__init__(**merged)


__all__ = ["Inkling2Config", "Inkling2TextConfig"]
