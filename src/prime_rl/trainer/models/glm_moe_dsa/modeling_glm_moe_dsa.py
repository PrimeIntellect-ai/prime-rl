import warnings
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import (
    RotaryEmbedding,
    RotaryEmbeddingConfig,
    apply_rotary_pos_emb_interleave,
)

try:
    from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
except ImportError:
    sparse_mla_fwd_interface = None  # type: ignore
    sparse_mla_bwd = None  # type: ignore

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer


class _SparseMLA(torch.autograd.Function):
    """Autograd wrapper for tilelang sparse MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, out, indices, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, out, indices, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(q, kv, out, do.contiguous(), indices, lse, sm_scale=ctx.sm_scale)
        return dq, dkv, None, None


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps).type_as(x)


class Indexer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        if config.q_lora_rank is None:
            raise ValueError("Sparse indexer requires q_lora_rank to be set")

        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=config.attention_bias)
        self.k_norm = LayerNorm(dim=self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_head, bias=False)
        self.weight_scale = (self.head_dim**-0.5) * (self.n_head**-0.5)

    @torch.no_grad()
    def compute_sparse_indices(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        index_topk: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[1]
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        q_idx = self.wq_b(q_latent[0]).view(total_tokens, self.n_head, self.head_dim)
        k_idx = self.k_norm(self.wk(hidden_states[0]))
        w = self.weights_proj(hidden_states[0])

        q_pe = q_idx[..., : self.rope_dim]
        q_nope = q_idx[..., self.rope_dim :]
        k_pe = k_idx[..., : self.rope_dim]
        k_nope = k_idx[..., self.rope_dim :]

        cos, sin = position_embeddings
        q_pe = q_pe.unsqueeze(0).transpose(1, 2)
        k_pe = k_pe.unsqueeze(0).unsqueeze(1)
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2).squeeze(0)
        k_pe = k_pe.squeeze(1).squeeze(0)

        q_idx = torch.cat([q_pe, q_nope], dim=-1)
        k_idx = torch.cat([k_pe, k_nope], dim=-1)

        indices = fp8_indexer(q_idx, k_idx, w, ks, ke, index_topk, self.weight_scale)
        return indices.view(1, total_tokens, 1, index_topk)


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=self.kv_lora_rank, eps=config.rms_norm_eps))
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.indexer = Indexer(config)
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None = None,
        ke: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, total_tokens, _ = hidden_states.shape

        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_full = self.q_b_proj(q_latent).view(batch_size, total_tokens, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed_normed = self.kv_a_layernorm(k_compressed)

        q_rope_r = q_rope.transpose(1, 2)
        k_rope_r = k_rope.unsqueeze(1)
        cos, sin = position_embeddings
        q_rope_r, k_rope_r = apply_rotary_pos_emb_interleave(q_rope_r, k_rope_r, cos, sin)
        q_rope = q_rope_r.transpose(1, 2)
        k_rope = k_rope_r.squeeze(1)

        indices = self.indexer.compute_sparse_indices(
            hidden_states, q_latent, ks, ke, self.config.index_topk, position_embeddings
        )

        # KV absorption
        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]

        q_absorbed = torch.einsum("bshd,hdk->bshk", q_nope, w_k_nope)

        sparse_q = torch.cat([q_absorbed, q_rope], dim=-1)
        sparse_kv = torch.cat([k_compressed_normed, k_rope], dim=-1).unsqueeze(2)

        # Sentinel zero-token so padded indices (= total_tokens) are in-bounds
        sentinel = torch.zeros(batch_size, 1, 1, sparse_kv.shape[-1], dtype=sparse_kv.dtype, device=sparse_kv.device)
        sparse_kv = torch.cat([sparse_kv, sentinel], dim=1)

        out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)

        # Un-absorb
        out = torch.einsum("bshk,hdk->bshd", out, w_v)

        out = out.reshape(batch_size, total_tokens, -1)
        return self.o_proj(out), None


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(config)

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
            use_grouped_mm=config.use_grouped_mm,
            moe_backend=getattr(config, "moe_backend", "grouped_mm"),
        )
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ks: Optional[torch.Tensor] = None,
        ke: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class GlmMoeDsaPreTrainedModel(PreTrainedModelPrimeRL):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


@auto_docstring
class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        self.rotary_emb = RotaryEmbedding(rotary_config)
        self.gradient_checkpointing = False

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        flat_position_ids = position_ids.view(-1)
        S = flat_position_ids.shape[0]
        ks = torch.arange(S, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(torch.int32)
        ke = torch.arange(1, S + 1, dtype=torch.int32, device=flat_position_ids.device)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        warnings.warn("GlmMoeDsaForCausalLM is experimental, higher trainer<->inference KL mismatch may be observed.")
        warnings.warn("`model.attn` is ignored, GlmMoeDsa uses only sparse attention.")

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        """
        assert use_cache is None, "use_cache is not supported for custom glm_moe_dsa for now"
        assert past_key_values is None, "past_key_values is not supported for custom glm_moe_dsa for now"

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["GlmMoeDsaConfig", "GlmMoeDsaPreTrainedModel", "GlmMoeDsaModel", "GlmMoeDsaForCausalLM"]
