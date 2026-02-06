# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Glm4MoeLiteConfig(PretrainedConfig):
    r"""
    Configuration class for GLM-4.7-Flash (glm4_moe_lite) model.

    This model uses Multi-head Latent Attention (MLA) with LoRA-style compression
    for queries and key-values, combined with a Mixture of Experts architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10240):
            Dimension of the dense MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of the MoE expert representations.
        num_hidden_layers (`int`, *optional*, defaults to 47):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 20):
            Number of key-value heads (for GQA).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.8):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            LoRA rank for key-value projections.
        q_lora_rank (`int`, *optional*, defaults to 768):
            LoRA rank for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of query/key heads without rotary position embeddings.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups per token.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of experts per token.
        norm_topk_prob (`bool`, *optional*, defaults to True):
            Whether to normalize the top-k probabilities.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to use KV cache.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Base period of RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration.
        attention_bias (`bool`, *optional*, defaults to False):
            Whether to use attention bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        rope_interleave (`bool`, *optional*, defaults to True):
            Whether to use interleaved RoPE.
        mlp_layer_types (`list`, *optional*):
            Pattern specifying dense vs sparse (MoE) layers per layer.
        use_grouped_mm (`bool`, *optional*, defaults to True):
            Whether to use grouped matrix multiplication for MoE.
    """

    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_shared_experts: int = 1,
        n_routed_experts: int = 64,
        routed_scaling_factor: float = 1.8,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 768,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 4,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_interleave: bool = True,
        mlp_layer_types: list | None = None,
        use_grouped_mm: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # MLA attention parameters
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.rope_interleave = rope_interleave

        # MoE parameters
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.use_grouped_mm = use_grouped_mm

        # Layer types (dense vs sparse)
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        # General parameters
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate RoPE config
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if not self.use_grouped_mm:
            warnings.warn("Not using grouped mm for MoE is very slow, should only be used for debugging")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
