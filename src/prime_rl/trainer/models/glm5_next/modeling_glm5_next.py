import warnings
from typing import Optional, Union

import torch
import torch.distributed as dist
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d as fla_causal_conv1d
from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.kda import chunk_kda
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import _index_cache_skip_topk
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs
from prime_rl.trainer.models.layers.activations import ActivationType
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

from .configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from .converting_glm5_next import conversion_chain


@torch.compiler.disable
def _fla_causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
    cp_context: FLACPContext,
) -> torch.Tensor:
    output, _ = fla_causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        cp_context=cp_context,
    )
    return output


def _sparse_mla_attention_args(config: Glm5NextTextConfig, layer_idx: int) -> SparseMlaAttentionArgs:
    if config.q_lora_rank is None:
        raise ValueError("Sparse MLA attention requires q_lora_rank to be set")
    return SparseMlaAttentionArgs(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        use_index_cache=getattr(config, "use_index_cache", False),
        skip_topk=_index_cache_skip_topk(config, layer_idx),
    )


def _glm5_activation(config: Glm5NextTextConfig) -> ActivationType:
    if config.swiglu_limit is None:
        return config.hidden_act
    if float(config.swiglu_limit) != 10.0:
        raise NotImplementedError("GLM-5.3 SwiGLU clamping only supports swiglu_limit=10.0")
    return "glm_clamped_swiglu"


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.projection_size = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        self.lower_bound = config.linear_lower_bound

        self.q_proj = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False)

        self.q_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            padding=self.conv_kernel_size - 1,
        )
        self.k_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            padding=self.conv_kernel_size - 1,
        )
        self.v_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            padding=self.conv_kernel_size - 1,
        )

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, self.hidden_size, bias=False)

        self.cp_group: dist.ProcessGroup | None = None
        self.cp_rank: int = 0
        self.cp_world_size: int = 1

    def _build_cp_context(
        self,
        device: torch.device,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_are_pre_shard: bool = False,
    ) -> FLACPContext | None:
        if self.cp_group is None:
            return None
        if cu_seqlens is None or not cu_seqlens_are_pre_shard:
            raise ValueError("GLM-5.3 KDA context parallelism requires full pre-shard sequence boundaries")
        return build_cp_context(
            cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
            group=self.cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )

    def _conv1d(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        cu_seqlens: torch.LongTensor | None,
        cp_context: FLACPContext | None,
    ) -> torch.Tensor:
        weight = conv.weight.squeeze(1)
        if cp_context is None:
            out, _ = fla_causal_conv1d(
                x=x,
                weight=weight,
                bias=conv.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            return out
        return _fla_causal_conv1d_cp(
            x=x,
            weight=weight,
            bias=conv.bias,
            activation=self.activation,
            cp_context=cp_context,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_are_pre_shard: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cp_context = self._build_cp_context(hidden_states.device, cu_seqlens, cu_seqlens_are_pre_shard)

        query = self._conv1d(self.q_proj(hidden_states), self.q_conv1d, cu_seqlens, cp_context)
        key = self._conv1d(self.k_proj(hidden_states), self.k_conv1d, cu_seqlens, cp_context)
        value = self._conv1d(self.v_proj(hidden_states), self.v_conv1d, cu_seqlens, cp_context)

        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        beta = self.b_proj(hidden_states)
        gate_a = self.f_a_proj(hidden_states)
        gate = self.f_b_proj(gate_a).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        core_attn_out, _ = chunk_kda(
            query,
            key,
            value,
            g=gate,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        core_attn_out = self.o_norm(core_attn_out, output_gate)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.projection_size)
        return self.o_proj(core_attn_out)


def _hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1]).contiguous()


def _hc_contract(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=-2)


def _mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    fn = fn.to(torch.float32)
    hc_scale = hc_scale.to(torch.float32)
    hc_base = hc_base.to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult)
    comb_logits = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1).to(residual.dtype)
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def _mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    return (mixed_residual + post_term).to(residual.dtype)


class Glm5NextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_type = config.glm5_layer_types[layer_idx]
        self.mhc = bool(config.mhc)

        if self.layer_type == "linear_attention":
            self.self_attn = Glm5NextLinearAttention(config)
        elif self.layer_type == "deepseek_sparse_attention":
            self.self_attn = GlmMoeDsaAttention(_sparse_mla_attention_args(config, layer_idx))
        else:
            raise ValueError(f"Unsupported GLM-5.3 layer_type {self.layer_type!r}")

        activation = _glm5_activation(config)
        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            expert_type="gated",
            activation=activation,
            score_func=config.scoring_func,
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=config.load_balance_coeff,
        )
        self.mlp_type = config.mlp_layer_types[layer_idx]
        if config.is_moe and self.mlp_type == "sparse":
            shared_expert = None
            if config.n_shared_experts > 0:
                shared_expert = FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size * config.n_shared_experts,
                    expert_type=moe_args.expert_type,
                    activation=moe_args.activation,
                )
            self.mlp = MoE.from_args(
                moe_args,
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                shared_expert=shared_expert,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                activation=activation,
            )

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        if self.mhc:
            self.n = config.mhc_num_residual_streams
            d_model = self.n * self.hidden_size
            mix_hc = (2 + self.n) * self.n
            self.hc_eps = config.hc_eps
            self.rms_norm_eps = config.rms_norm_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        if self.layer_type == "linear_attention":
            self.self_attn.cp_group = cp_group
            self.self_attn.cp_rank = cp_rank
            self.self_attn.cp_world_size = cp_world_size
        else:
            self.self_attn.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _run_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None,
        ke: torch.Tensor | None,
        cached_indices: torch.Tensor | None,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_are_pre_shard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.layer_type == "linear_attention":
            return (
                self.self_attn(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_are_pre_shard=cu_seqlens_are_pre_shard,
                ),
                cached_indices,
            )
        return self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
            cached_indices=cached_indices,
        )

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _mhc_pre_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
        )

    def _forward_without_mhc(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None,
        ke: torch.Tensor | None,
        cached_indices: torch.Tensor | None,
        routed_experts: Optional[torch.LongTensor],
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_are_pre_shard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cached_indices = self._run_attention(
            hidden_states,
            position_embeddings,
            ks,
            ke,
            cached_indices,
            cu_seqlens,
            cu_seqlens_are_pre_shard,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        hidden_states = residual + hidden_states
        return hidden_states, cached_indices, None, None, None

    def _forward_with_mhc(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None,
        ke: torch.Tensor | None,
        cached_indices: torch.Tensor | None,
        routed_experts: Optional[torch.LongTensor],
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_are_pre_shard: bool,
        hc_residual: torch.Tensor | None,
        hc_post: torch.Tensor | None,
        hc_comb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        x = hidden_states
        if hc_post is None:
            if hc_residual is None:
                hc_residual = _hc_expand(x, self.n)
            hc_post, hc_comb, x = self._hc_pre(hc_residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        else:
            hc_residual = _mhc_post_torch(x, hc_residual, hc_post, hc_comb)
            hc_post, hc_comb, x = self._hc_pre(hc_residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)

        x = self.input_layernorm(x)
        x, cached_indices = self._run_attention(
            x,
            position_embeddings,
            ks,
            ke,
            cached_indices,
            cu_seqlens,
            cu_seqlens_are_pre_shard,
        )

        hc_residual = _mhc_post_torch(x, hc_residual, hc_post, hc_comb)
        hc_post, hc_comb, x = self._hc_pre(hc_residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, routed_experts=routed_experts)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = _mhc_post_torch(x, hc_residual, hc_post, hc_comb)
            return _hc_contract(x), cached_indices, None, None, None
        return x, cached_indices, hc_residual, hc_post, hc_comb

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None = None,
        ke: torch.Tensor | None = None,
        cached_indices: torch.Tensor | None = None,
        routed_experts: Optional[torch.LongTensor] = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_are_pre_shard: bool = False,
        hc_residual: torch.Tensor | None = None,
        hc_post: torch.Tensor | None = None,
        hc_comb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.mhc:
            return self._forward_without_mhc(
                hidden_states,
                position_embeddings,
                ks,
                ke,
                cached_indices,
                routed_experts,
                cu_seqlens,
                cu_seqlens_are_pre_shard,
            )
        return self._forward_with_mhc(
            hidden_states,
            position_embeddings,
            ks,
            ke,
            cached_indices,
            routed_experts,
            cu_seqlens,
            cu_seqlens_are_pre_shard,
            hc_residual,
            hc_post,
            hc_comb,
        )


class Glm5NextPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Glm5NextTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm5NextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Glm5NextDecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        text_config = self.config.get_text_config() if isinstance(self.config, Glm5NextConfig) else self.config
        std = getattr(text_config, "initializer_range", 0.02)
        if isinstance(module, MoE):
            module.init_weights(std, buffer_device=module.tokens_per_expert.device)
        elif isinstance(module, Glm5NextLinearAttention):
            nn.init.zeros_(module.A_log)
            nn.init.zeros_(module.dt_bias)
        elif isinstance(module, Glm5NextDecoderLayer) and module.mhc:
            for weight in (module.hc_attn_fn, module.hc_ffn_fn):
                nn.init.trunc_normal_(weight, mean=0.0, std=std)
            for bias in (module.hc_attn_base, module.hc_ffn_base):
                nn.init.zeros_(bias)
            for scale in (module.hc_attn_scale, module.hc_ffn_scale):
                nn.init.ones_(scale)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(("mlp.router.selection_bias", "self_attn.A_log", "self_attn.dt_bias")) or ".hc_" in name

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "model.language_model.layers." in name
            or "mlp.experts.1.up_proj" in name
            or "mlp.experts.gate_up_proj" in name
            or "self_attn.indexer.index_kpool_compress_" in name
            for name in state_dict.keys()
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.startswith("model.") for name in state_dict.keys()) and not any(
            name.startswith("model.language_model.") for name in state_dict.keys()
        )

    @classmethod
    def conversion_chain(cls, config):
        text_config = config.get_text_config() if isinstance(config, Glm5NextConfig) else config
        return conversion_chain(text_config)


class Glm5NextModel(Glm5NextPreTrainedModel):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm5NextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = self._create_rotary_emb(config)
        self.gradient_checkpointing = False
        self.post_init()

    def _create_rotary_emb(self, config: Glm5NextTextConfig) -> RotaryEmbedding | None:
        if config.qk_rope_head_dim == 0:
            return None
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        return RotaryEmbedding(rotary_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        for layer in self.layers:
            layer.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _context_parallel_state(self) -> tuple[dist.ProcessGroup | None, int, int]:
        if len(self.layers) == 0:
            return None, 0, 1
        layer = self.layers[0]
        return getattr(layer, "_cp_group", None), getattr(layer, "_cp_rank", 0), getattr(layer, "_cp_world_size", 1)

    def _gather_position_ids_for_cp(
        self,
        position_ids: torch.LongTensor,
        cp_group: dist.ProcessGroup,
        cp_world_size: int,
    ) -> torch.LongTensor:
        gathered_position_ids = [torch.empty_like(position_ids) for _ in range(cp_world_size)]
        dist.all_gather(gathered_position_ids, position_ids.contiguous(), group=cp_group)
        return torch.cat(gathered_position_ids, dim=1)

    def _position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids_full: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_emb is None:
            shape = (*position_ids_full.shape, 0)
            empty = hidden_states.new_empty(shape)
            return empty, empty
        return self.rotary_emb(hidden_states, position_ids_full)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        *,
        seq_lens: Optional[torch.LongTensor] = None,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if seq_lens is None:
            seq_lens = torch.tensor([inputs_embeds.shape[1]], dtype=torch.long, device=inputs_embeds.device)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        del max_seqlen
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        cp_group, cp_rank, cp_world_size = self._context_parallel_state()
        if cp_group is not None and cp_world_size > 1:
            position_ids_full = self._gather_position_ids_for_cp(position_ids, cp_group, cp_world_size)
        else:
            position_ids_full = position_ids

        flat_position_ids = position_ids_full.reshape(-1)
        s_full = flat_position_ids.shape[0]
        ks_full = torch.arange(s_full, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(
            torch.int32
        )
        ke_full = torch.arange(1, s_full + 1, dtype=torch.int32, device=flat_position_ids.device)

        hidden_states = inputs_embeds
        position_embeddings = self._position_embeddings(hidden_states, position_ids_full)

        if cp_world_size > 1:
            s_local = s_full // cp_world_size
            ks = ks_full[cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
            ke = ke_full[cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
        else:
            ks, ke = ks_full, ke_full

        cached_indices = None
        hc_residual = None
        hc_post = None
        hc_comb = None
        use_index_cache = getattr(self.config, "use_index_cache", False)
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states, next_cached_indices, hc_residual, hc_post, hc_comb = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
                cached_indices=cached_indices,
                routed_experts=routed_experts_layer,
                cu_seqlens=cu_seqlens,
                cu_seqlens_are_pre_shard=seq_lens_are_pre_shard,
                hc_residual=hc_residual,
                hc_post=hc_post,
                hc_comb=hc_comb,
            )
            cached_indices = next_cached_indices if use_index_cache else None

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Glm5NextForCausalLM(Glm5NextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        text_config = config.get_text_config() if isinstance(config, Glm5NextConfig) else config
        super().__init__(config, **kwargs)
        self.model = Glm5NextModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        warnings.warn(
            "Glm5NextForCausalLM is experimental: KDA and mHC use correctness-first training paths, "
            "and the sparse MLA indexer does not yet implement GLM-5.3 k-pool compression."
        )
        warnings.warn("`model.attn` is ignored, GLM-5.3 dispatches per-layer KDA/sparse MLA from layer_types.")

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self.model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        del attention_mask, cache_position, kwargs
        assert use_cache is None, "use_cache is not supported for custom glm5_next for now"
        assert past_key_values is None, "past_key_values is not supported for custom glm5_next for now"

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        if self.model.rotary_emb is None:
            return
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


__all__ = [
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextPreTrainedModel",
    "Glm5NextModel",
    "Glm5NextForCausalLM",
]
