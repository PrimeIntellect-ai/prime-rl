# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 model configs."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers.configuration_utils import PreTrainedConfig

LOOP_CONFIG_KEYS = {"core_start", "core_end", "num_passes", "gate_rank", "is_varloop"}


class Inkling2ModelConfig(PreTrainedConfig):
    model_type = "inkling2_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 1024,
        sconv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        model_max_length: int = 1048576,
        log_scaling_n_floor: int | None = 128000,
        log_scaling_alpha: float = 0.1,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        final_logit_softcapping: float | None = 30.0,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 2304,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        loop_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if pre_norm_pre_affine or final_norm_pre_affine:
            raise ValueError("Inkling2 pre-affine norms are not supported")
        if loop_config is not None and (unknown := set(loop_config) - LOOP_CONFIG_KEYS):
            raise ValueError(f"Unsupported Inkling2 loop_config keys {sorted(unknown)}")
        # Looped checkpoints (Inkling 2 Mini) run layers [core_start, core_end)
        # num_passes times with shared weights.
        self.loop_config = dict(loop_config) if loop_config is not None else None
        if "n_layers_per_attnres_block" in kwargs:
            raise ValueError("n_layers_per_attnres_block was renamed to n_layers_per_attn_res_block")
        self.padded_vocab_size = vocab_size
        self.vocab_size = unpadded_vocab_size or vocab_size
        self.unpadded_vocab_size = self.vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads or num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads or num_key_value_heads
        self.swa_head_dim = swa_head_dim or head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        if local_layer_ids is None:
            local_layer_ids = [
                layer for layer in range(num_hidden_layers) if (layer + 1) % n_layers_per_attn_res_block != 0
            ]
        self.local_layer_ids = list(local_layer_ids)
        self.sliding_window_size = sliding_window_size
        self.sconv_kernel_size = sconv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.final_logit_softcapping = final_logit_softcapping
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def loop_core_layers(self) -> range:
        """Layers that run once per loop pass (none without a loop)."""
        if self.loop_config is None:
            return range(0)
        return range(self.loop_config["core_start"], self.loop_config["core_end"])

    @property
    def num_loop_passes(self) -> int:
        return 1 if self.loop_config is None else self.loop_config["num_passes"]

    @property
    def num_layer_visits(self) -> int:
        """Layer calls of one forward; the routed experts hold one row per visit."""
        return self.num_hidden_layers + (self.num_loop_passes - 1) * len(self.loop_core_layers)

    def layer_visit_ids(self, layer_id: int) -> tuple[int, ...]:
        """The visits of a layer, one per pass it runs in, in forward order."""
        core = self.loop_core_layers
        if layer_id in core:
            return tuple(layer_id + pass_idx * len(core) for pass_idx in range(self.num_loop_passes))
        if layer_id < core.start:
            return (layer_id,)
        return (layer_id + (self.num_loop_passes - 1) * len(core),)

    def check_loop_options(self) -> None:
        """Validate the loop against the layers; `hf_overrides` change them after the constructor."""
        if self.loop_config is None:
            return
        if self.loop_config.get("is_varloop", False):
            raise ValueError("Inkling2 checkpoints with a variable number of loop passes are not supported")
        core, block = self.loop_core_layers, self.n_layers_per_attn_res_block
        if not 0 < core.start < core.stop <= self.num_hidden_layers:
            raise ValueError(
                f"The loop core [{core.start}, {core.stop}) must start after layer 0 and end within the "
                f"{self.num_hidden_layers} layers of the model (a truncated model must keep the whole core)"
            )
        # A pass restarts from the block the first core layer seals, and its
        # messages read the blocks of the previous pass: at least two.
        if core.start % block or core.stop % block or len(core) < 2 * block:
            raise ValueError(
                f"The loop core [{core.start}, {core.stop}) must cover two or more blocks of {block} layers"
            )
        if self.num_loop_passes < 1 or self.loop_config["gate_rank"] < 1:
            raise ValueError("Inkling2 loop num_passes and gate_rank must be at least 1")
        if not self.attn_res_normalize_values:
            raise ValueError("Inkling2 loop messages need attn_res_normalize_values")


class Inkling2MMConfig(PreTrainedConfig):
    model_type = "inkling2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Whether the checkpoint has a loop: `inkling2_loop` checkpoints do.
    looped: ClassVar[bool] = False
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "text_config": Inkling2ModelConfig,
    }

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2ModelConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config if isinstance(text_config, Inkling2ModelConfig) else Inkling2ModelConfig(**(text_config or {}))
        )
        self.vision_config = vision_config
        if (self.text_config.loop_config is not None) != self.looped:
            raise ValueError(
                f"Inkling2 model_type {self.model_type!r} {'needs' if self.looped else 'does not take'} a loop_config: "
                "looped checkpoints are model_type 'inkling2_loop'"
            )
        # The checkpoint's unembedding runs in BF16 with an FP32 output.
        kwargs.setdefault("head_dtype", "float32")
        kwargs.setdefault("torch_dtype", "bfloat16")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # Top-level aliases vLLM reads off the config.
        self.vocab_size = self.text_config.vocab_size
        self.max_position_embeddings = self.text_config.max_position_embeddings

    def get_text_config(self, *args: Any, **kwargs: Any) -> Inkling2ModelConfig:
        return self.text_config


class Inkling2LoopMMConfig(Inkling2MMConfig):
    """Looped Inkling2 checkpoints (Inkling 2 Mini)."""

    model_type = "inkling2_loop"
    looped: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Inkling2DSparkConfig(PreTrainedConfig):
    """The DSpark draft of an Inkling2 checkpoint, from its `mtp_config`."""

    model_type = "inkling2_dspark"

    def __init__(
        self,
        *,
        block_size: int = 8,
        target_layer_ids: list[int] | None = None,
        **kwargs,
    ) -> None:
        # Not stored as ``block_size``: vLLM DSpark would take that as the
        # number of speculative tokens, but the block holds an anchor plus
        # block_size - 1 masks, each predicting the token at its own position.
        self.draft_block_size = block_size
        self.dspark_block_size = block_size - 1
        self.sample_from_anchor = False
        self.is_causal = False
        self.mask_token_id = 0
        # Target layers whose input (the AttnRes partial block) is captured; the
        # last id, the target's num_hidden_layers, is its final-norm output.
        self.target_layer_ids = list(target_layer_ids or [])
        self.eagle_aux_hidden_state_layer_ids = self.target_layer_ids
        kwargs.setdefault("architectures", ["Inkling2DSparkModel"])
        super().__init__(tie_word_embeddings=False, **kwargs)

    @classmethod
    def from_target(cls, config: Inkling2MMConfig) -> Inkling2DSparkConfig:
        mtp_config = getattr(config, "mtp_config", None)
        if not mtp_config:
            raise ValueError("The Inkling2 checkpoint has no mtp_config")
        text = config.text_config
        draft = {k: v for k, v in mtp_config.items() if k not in ("torch_dtype", "dtype", "architectures")}
        features = draft.get("target_layer_ids", [])
        if len(features) < 2 or features != sorted(set(features)) or features[-1] != text.num_hidden_layers:
            raise ValueError(
                "Inkling2 MTP target_layer_ids must be sorted target layer ids "
                "followed by num_hidden_layers (the final-norm output)"
            )
        if (
            draft["hidden_size"] != text.hidden_size
            or draft.get("markov_head_type") != "rnn_hidden"
            or draft.get("q_bias")
            or draft.get("o_bias")
        ):
            raise ValueError("Unsupported Inkling2 mtp_config")
        return cls(
            **draft,
            vocab_size=text.vocab_size,
            padded_vocab_size=text.padded_vocab_size,
            final_logit_softcapping=text.final_logit_softcapping,
            torch_dtype="bfloat16",
        )
