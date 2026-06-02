import math
import re
from copy import copy
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
    Gemma4CausalLMOutputWithPast,
    Gemma4ClippableLinear,
    Gemma4ModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionRotaryEmbedding,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
)
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from typing_extensions import Unpack

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise
from prime_rl.trainer.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from prime_rl.trainer.models.layers.attn import AttentionConfig, SDPAAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import GemmaGroupedExperts, TokenReorderer
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig, RMSNormNoScale
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb
from prime_rl.utils.sequence_packing import infer_cu_seqlens_from_position_ids

_CLIPPABLE_BUFFER_NAMES = frozenset({"input_min", "input_max", "output_min", "output_max"})


# ----------------------------------------------------------------------------- #
# RoPE config helpers (per-layer-type). gemma-4 uses a "proportional" rope on the
# global (full_attention) layers whose head_dim is `global_head_dim` (512), and a
# plain "default" rope on the sliding layers (head_dim 256). The global rope MUST
# be initialized with head_dim=512 or the inv_freq (and thus the rotated dims) are
# wrong — that is the root cause of the layer-5/11/17 grad-norm blowup (#2362).
# ----------------------------------------------------------------------------- #
def _prime_attn_impl(config: Gemma4TextConfig) -> str:
    attn_impl = config._attn_implementation
    return "fa4" if attn_impl == "flash_attention_4" else attn_impl


def _rope_type(rope_config: dict) -> str:
    return rope_config.get("rope_type", rope_config.get("type", "default"))


def _layer_rope_config(config: Gemma4TextConfig, layer_type: str) -> dict:
    rope_config = getattr(config, "rope_parameters", None)
    if isinstance(rope_config, dict) and isinstance(rope_config.get(layer_type), dict):
        return rope_config[layer_type]
    raise ValueError(f"gemma-4 expects a per-layer-type rope_parameters dict; missing entry for {layer_type!r}")


def _rotary_embedding_config(config: Gemma4TextConfig, layer_type: str) -> RotaryEmbeddingConfig:
    rope_config = _layer_rope_config(config, layer_type)
    rotary_model_config = copy(config)
    rotary_model_config.rope_parameters = rope_config
    rotary_model_config.rope_scaling = rope_config
    # #2362 fix: the global ("full_attention") rope must see head_dim=512 so the
    # proportional init produces 64 rotated + 192 zero-padded inv_freqs (→ 512 cos/sin).
    if layer_type != "sliding_attention" and config.global_head_dim:
        rotary_model_config.head_dim = config.global_head_dim
    else:
        rotary_model_config.head_dim = config.head_dim
    return RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type=_rope_type(rope_config),
        model_config=rotary_model_config,
    )


def _sliding_window_mask(q_len: int, k_len: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # The additive mask MUST share the query/key/value dtype: SDPA's fused GPU kernels produce
    # NaN when handed an fp32 `-inf` bias against bf16 q/k/v (the CPU math backend silently
    # upcasts and hides this). Build the sentinel in `dtype` so every backend agrees.
    query_positions = torch.arange(q_len, device=device).unsqueeze(1)
    key_positions = torch.arange(k_len, device=device).unsqueeze(0)
    allowed = (key_positions <= query_positions) & (key_positions >= query_positions - window + 1)
    mask = torch.full((q_len, k_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    return mask.masked_fill(allowed, 0.0)


# ----------------------------------------------------------------------------- #
# Attention. gemma-4 deltas vs the generic SDPAAttention:
#   - scaling = 1.0 (qk-norm regime), not head_dim**-0.5
#   - per-head q_norm/k_norm AND a no-scale v_norm
#   - `attention_k_eq_v` on global layers: no v_proj; V derives from the viewed
#     (pre-k_norm) k_proj output, gets v_norm, and is NOT rotated
# ----------------------------------------------------------------------------- #
class Gemma4SDPAAttention(SDPAAttention):
    def __init__(self, config: AttentionConfig, use_alternative_attention: bool = False):
        super().__init__(config)
        self.use_alternative_attention = use_alternative_attention
        self.v_norm = RMSNormNoScale(config.rms_norm_eps)
        if use_alternative_attention:
            # k_eq_v: value is derived from k_proj, so there is no v_proj weight.
            self.v_proj = None

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        # V from the pre-norm k_proj output (k_eq_v) or its own v_proj. Bind before k_norm.
        value_states = key_states if self.use_alternative_attention else self.v_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def _attention_core_single_sequence(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is None:
            out = F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True, scale=self.scaling
            )
        else:
            mask = _sliding_window_mask(
                query_states.shape[-2], key_states.shape[-2], sliding_window, query_states.device, query_states.dtype
            )
            out = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=mask, scale=self.scaling
            )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return self._attention_core_single_sequence(query_states, key_states, value_states)

        if query_states.shape[0] != 1:
            raise ValueError("Packed SDPA reference path expects batch size 1")

        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            outputs.append(
                self._attention_core_single_sequence(
                    query_states[:, :, start:end, :],
                    key_states[:, :, start:end, :],
                    value_states[:, :, start:end, :],
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(
            query_states, key_states, value_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        attn_output = self.output_proj(attn_output)
        return attn_output, None


def _check_unsupported(config: Gemma4TextConfig) -> None:
    if config.num_kv_shared_layers:
        raise NotImplementedError("gemma-4 KV-sharing layers are not supported in the prime-rl port")
    if config.use_double_wide_mlp:
        raise NotImplementedError("gemma-4 double-wide MLP is not supported in the prime-rl port")
    if config.hidden_size_per_layer_input:
        raise NotImplementedError("gemma-4 per-layer-input is not supported in the prime-rl port")


def _check_supported_config(config: Gemma4TextConfig) -> None:
    _check_unsupported(config)
    attn_impl = _prime_attn_impl(config)
    if attn_impl != "sdpa":
        global_head_dim = config.global_head_dim or config.head_dim
        raise ValueError(
            f"gemma-4 requires attn='sdpa': its head_dim={global_head_dim} global attention layers have no flash "
            f"kernel; got attn={attn_impl!r}"
        )


def _build_gemma_self_attn(config: Gemma4TextConfig, layer_idx: int) -> "Gemma4SDPAAttention":
    is_sliding = config.layer_types[layer_idx] == "sliding_attention"
    head_dim = config.head_dim if is_sliding else (config.global_head_dim or config.head_dim)
    use_alternative_attention = bool(config.attention_k_eq_v) and not is_sliding
    num_key_value_heads = config.num_global_key_value_heads if use_alternative_attention else config.num_key_value_heads

    attn_config = AttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        is_causal=True,
        attention_bias=config.attention_bias,
        use_qk_norm=True,
        rms_norm_eps=config.rms_norm_eps,
        qk_norm_type="per_head",
        scaling=1.0,
    )
    self_attn = Gemma4SDPAAttention(attn_config, use_alternative_attention=use_alternative_attention)
    self_attn.sliding_window = config.sliding_window if is_sliding else None
    return self_attn


class Gemma4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = _build_gemma_self_attn(config, layer_idx)

        self.mlp = MLP(
            MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_activation,
                bias=False,
            )
        )
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.pre_feedforward_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_feedforward_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        # Persistent per-layer output scalar (loaded from checkpoint; ones at init).
        self.register_buffer("layer_scalar", torch.ones(1))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar.to(hidden_states.dtype)
        return hidden_states


class Gemma4TextRouter(nn.Module):
    """gemma-4 MoE router. Param names/layout mirror HF ``Gemma4TextRouter`` (identity checkpoint load):
    RMSNorm-no-scale → per-dim ``scale`` × ``hidden**-0.5`` → softmax over ALL experts → top-k → renorm
    to sum-1 → × ``per_expert_scale[idx]``. No aux/load-balance loss. Returns ``(top_k_weights, top_k_index)``;
    the full softmax (HF's first return) is unused downstream.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.top_k = config.top_k_experts
        self.scalar_root_size = config.hidden_size**-0.5
        self.norm = RMSNormNoScale(config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        router_probabilities = F.softmax(expert_scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


class Gemma4MoEDecoderLayer(GradientCheckpointingLayer):
    """gemma-4 hybrid dual-path decoder layer (``enable_moe_block=True``). A dense shared ``mlp`` and
    128 routed ``experts`` BOTH read the pre-MLP residual through separate norms; their separately-normed
    outputs are summed and re-normed. 7 norms/layer. Mirrors HF ``Gemma4TextDecoderLayer`` forward.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = _build_gemma_self_attn(config, layer_idx)

        self.mlp = MLP(
            MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_activation,
                bias=False,
            )
        )
        self.router = Gemma4TextRouter(config)
        self.experts = GemmaGroupedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            use_grouped_mm=getattr(config, "use_grouped_mm", True),
            fp8=getattr(config, "fp8", False),
        )
        self.reorderer = TokenReorderer(num_experts=config.num_experts, top_k=config.top_k_experts)

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.pre_feedforward_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.post_feedforward_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.post_feedforward_layernorm_1 = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.pre_feedforward_layernorm_2 = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        self.post_feedforward_layernorm_2 = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=eps))
        # Persistent per-layer output scalar (loaded from checkpoint; ones at init).
        self.register_buffer("layer_scalar", torch.ones(1))

    def _run_experts(self, residual: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = residual.shape
        # Router reads the RAW residual; experts read a *separate* pre-norm of the same residual.
        top_scores, top_indices = self.router(residual.reshape(-1, dim))
        x = self.pre_feedforward_layernorm_2(residual).reshape(-1, dim)

        # Sort the (token, expert) pairs by expert id; histc gives per-expert token counts.
        top_scores_sorted, token_indices_sorted, num_tokens_per_expert = self.reorderer(top_scores, top_indices)

        gather_indices = token_indices_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=gather_indices)
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        # Routing weight multiplies the expert output (HF: ``out *= top_k_weights``), i.e. score-after-experts.
        routed_output = (routed_output.float() * top_scores_sorted.reshape(-1, 1)).to(routed_output.dtype)

        out = torch.zeros_like(x)
        out = out.scatter_add(dim=0, index=gather_indices, src=routed_output)
        return out.reshape(bs, slen, dim)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        mlp_out = self.post_feedforward_layernorm_1(self.mlp(self.pre_feedforward_layernorm(residual)))
        experts_out = self.post_feedforward_layernorm_2(self._run_experts(residual))
        hidden_states = self.post_feedforward_layernorm(mlp_out + experts_out)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar.to(hidden_states.dtype)
        return hidden_states


def _add_maybe_fp8(out: dict[str, Tensor], name: str, tensor: Tensor, quantize_fp8: bool) -> None:
    if quantize_fp8 and tensor.ndim == 2:
        fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
        out[name] = fp8_weight
        out[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
        return
    out[name] = tensor


def _convert_gemma4_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    prefix = f"model.layers.{layer_idx}"

    for name in [
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.pre_feedforward_layernorm.weight",
        f"{prefix}.post_feedforward_layernorm.weight",
        # MoE-only norms (guarded by membership; absent on dense layers).
        f"{prefix}.post_feedforward_layernorm_1.weight",
        f"{prefix}.pre_feedforward_layernorm_2.weight",
        f"{prefix}.post_feedforward_layernorm_2.weight",
        f"{prefix}.layer_scalar",
    ]:
        if name in state_dict:
            out[name] = state_dict[name]

    q_key = f"{prefix}.self_attn.q_proj.weight"
    k_key = f"{prefix}.self_attn.k_proj.weight"
    v_key = f"{prefix}.self_attn.v_proj.weight"
    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        qkv = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
        _add_maybe_fp8(out, f"{prefix}.self_attn.qkv_proj.weight", qkv, quantize_fp8)
    else:
        # k_eq_v global layers have no v_proj: keep q/k separate, vLLM derives v from k.
        if q_key in state_dict:
            _add_maybe_fp8(out, q_key, state_dict[q_key], quantize_fp8)
        if k_key in state_dict:
            _add_maybe_fp8(out, k_key, state_dict[k_key], quantize_fp8)

    o_key = f"{prefix}.self_attn.o_proj.weight"
    if o_key in state_dict:
        _add_maybe_fp8(out, o_key, state_dict[o_key], quantize_fp8)

    gate_key = f"{prefix}.mlp.gate_proj.weight"
    up_key = f"{prefix}.mlp.up_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        gate_up = torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0)
        _add_maybe_fp8(out, f"{prefix}.mlp.gate_up_proj.weight", gate_up, quantize_fp8)

    down_key = f"{prefix}.mlp.down_proj.weight"
    if down_key in state_dict:
        _add_maybe_fp8(out, down_key, state_dict[down_key], quantize_fp8)

    # MoE layers: router params + per-expert fused weights pass through as identity (HF layout).
    # NOTE: the gemma-4 MoE vLLM *inference* kernel mapping is UNVERIFIED — no A4B checkpoint is
    # available to test against. We emit a COMPLETE key set so a layout mismatch fails loudly at
    # load rather than silently dropping the expert weights.
    for name in [
        f"{prefix}.router.proj.weight",
        f"{prefix}.router.scale",
        f"{prefix}.router.per_expert_scale",
        f"{prefix}.experts.gate_up_proj",
        f"{prefix}.experts.down_proj",
    ]:
        if name in state_dict:
            out[name] = state_dict[name]

    return out


# The gemma-4 VLM checkpoint stores the text stack under `model.language_model.*` and the
# modality towers under `model.vision_tower.*` / `model.embed_vision.*`. The native text
# model is flat (`model.embed_tokens`, `model.layers.N`, `model.norm`). convert_to_prime
# strips the `language_model.` prefix and drops the (unused, for text-only RL) towers;
# convert_to_hf restores the VLM layout so saved/broadcast weights round-trip to vLLM/HF.
_VLM_TEXT_PREFIX = "model.language_model."
_PRIME_TEXT_PREFIXES = ("model.embed_tokens", "model.layers", "model.norm")
_DROP_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
    "model.audio_tower.",
    "model.multi_modal_projector.",
)


def _vlm_key_to_prime(key: str) -> str | None:
    if key.startswith(_VLM_TEXT_PREFIX):
        return "model." + key[len(_VLM_TEXT_PREFIX) :]
    if key.startswith("language_model."):
        return "model." + key[len("language_model.") :]
    if any(key.startswith(p) for p in _DROP_PREFIXES):
        return None
    return key


def _prime_key_to_vlm(key: str) -> str:
    if any(key.startswith(p) for p in _PRIME_TEXT_PREFIXES):
        return _VLM_TEXT_PREFIX + key[len("model.") :]
    return key


class Gemma4PreTrainedModel(PreTrainedModelPrimeRL):
    config: Gemma4TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4DecoderLayer", "Gemma4MoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Gemma4DecoderLayer,
    }

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(k.startswith(_VLM_TEXT_PREFIX) or k.startswith("language_model.") for k in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(k.startswith("model.layers.") for k in state_dict) and not cls.is_hf_state_dict(state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {_prime_key_to_vlm(k): v for k, v in state_dict.items()}

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {pk: v for k, v in state_dict.items() if (pk := _vlm_key_to_prime(k)) is not None}

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return {_prime_key_to_vlm(k): v for k, v in state_dict.items()}

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return {pk: v for k, v in state_dict.items() if (pk := _vlm_key_to_prime(k)) is not None}

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls, state_dict: dict[str, Tensor], layer_idx: int, quantize_fp8: bool = False
    ) -> dict[str, Tensor]:
        kernel = _convert_gemma4_layer_to_vllm_kernel(state_dict, layer_idx, quantize_fp8=quantize_fp8)
        return {_prime_key_to_vlm(k): v for k, v in kernel.items()}


class Gemma4Model(Gemma4PreTrainedModel):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        _check_supported_config(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.hidden_size**0.5
        )
        layer_cls = Gemma4MoEDecoderLayer if config.enable_moe_block else Gemma4DecoderLayer
        self.layers = nn.ModuleList([layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.rotary_embs = nn.ModuleDict(
            {
                layer_type: RotaryEmbedding(_rotary_embedding_config(config, layer_type))
                for layer_type in dict.fromkeys(config.layer_types)
            }
        )
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(inputs_embeds.shape[0], -1)
            )

        if (cu_seqlens is None) != (max_seqlen is None):
            raise ValueError("cu_seqlens and max_seqlen must be provided together")

        if cu_seqlens is None and inputs_embeds.shape[0] == 1:
            cu_seqlens, max_seqlen = infer_cu_seqlens_from_position_ids(position_ids)

        hidden_states = inputs_embeds
        position_embeddings_by_type = {
            layer_type: rotary_emb(hidden_states, position_ids) for layer_type, rotary_emb in self.rotary_embs.items()
        }

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings_by_type[self.config.layer_types[layer_idx]],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)

        self.model = Gemma4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
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
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None or use_cache is False, "use_cache is not supported for custom Gemma4"
        assert past_key_values is None, "past_key_values is not supported for custom Gemma4"

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        for rotary_emb in self.model.rotary_embs.values():
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)
            rotary_emb.original_inv_freq = rotary_emb.inv_freq
        for module in self.modules():
            if isinstance(module, Gemma4TextScaledWordEmbedding):
                module.embed_scale.fill_(module.scalar_embed_scale)


# ----------------------------------------------------------------------------- #
# VLM (vision/audio) wrapper. The 31B checkpoint architecture is
# `Gemma4ForConditionalGeneration`; text-only RL routes through its language
# model. Multimodal towers stay on HF (unused for text RL); only the text stack
# above is the native prime-rl port. Wiring the VLM checkpoint onto the native
# text model is the remaining integration step (see report).
# ----------------------------------------------------------------------------- #
def _init_text_rotary_embedding(module: Gemma4TextRotaryEmbedding) -> None:
    for layer_type, rope_init_fn in module.rope_init_fns.items():
        rope_init_fn_kwargs = {"device": getattr(module, f"{layer_type}_inv_freq").device, "layer_type": layer_type}
        if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
            rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

        inv_freq, attention_scaling = rope_init_fn(module.config, **rope_init_fn_kwargs)
        getattr(module, f"{layer_type}_inv_freq").copy_(inv_freq)
        getattr(module, f"{layer_type}_original_inv_freq").copy_(inv_freq)
        setattr(module, f"{layer_type}_attention_scaling", attention_scaling)


def _init_vision_rotary_embedding(module: Gemma4VisionRotaryEmbedding) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_init_fn = (
        ROPE_INIT_FUNCTIONS[module.rope_type]
        if module.rope_type != "default"
        else module.compute_default_rope_parameters
    )
    inv_freq, attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
    module.inv_freq.copy_(inv_freq)
    module.original_inv_freq.copy_(inv_freq)
    module.attention_scaling = attention_scaling


def _init_audio_relative_position(module: Gemma4AudioRelPositionalEncoding) -> None:
    min_timescale = 1.0
    max_timescale = 10000.0
    num_timescales = module.hidden_size // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, device=module.inv_timescales.device) * -log_timescale_increment
    )
    module.inv_timescales.copy_(inv_timescales.unsqueeze(0).unsqueeze(0))


def _init_gemma4_nonpersistent_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Gemma4TextRotaryEmbedding):
            _init_text_rotary_embedding(module)
        elif isinstance(module, Gemma4VisionRotaryEmbedding):
            _init_vision_rotary_embedding(module)
        elif isinstance(module, Gemma4TextScaledWordEmbedding):
            module.embed_scale.fill_(module.scalar_embed_scale)
        elif isinstance(module, Gemma4AudioRelPositionalEncoding):
            _init_audio_relative_position(module)
        elif isinstance(module, Gemma4AudioAttention):
            module.softcap.fill_(module.attention_logits_soft_cap)
        elif isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            module.input_min.fill_(-float("inf"))
            module.input_max.fill_(float("inf"))
            module.output_min.fill_(-float("inf"))
            module.output_max.fill_(float("inf"))


def _mark_gemma4_runtime_buffers_nonpersistent(model: torch.nn.Module) -> None:
    """Match Gemma4 Hub checkpoints: clipping ranges are runtime constants."""
    for module in model.modules():
        if isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            module._non_persistent_buffers_set.update(_CLIPPABLE_BUFFER_NAMES)


class _Gemma4VLMPrimeRLMixin:
    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            out[re.sub(r"(\.layers\.\d+)\.experts\.", r"\1.moe.experts.", key)] = value
        return out

    def _prime_rl_forward_backbone(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast | Gemma4ModelOutputWithPast:
        model_kwargs = dict(kwargs)
        model_kwargs.pop("return_dict", None)
        if pixel_values is None and pixel_values_videos is None and input_features is None:
            return self.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                **model_kwargs,
            )
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **model_kwargs,
        )

    def init_buffers_post_meta(self) -> None:
        _init_gemma4_nonpersistent_buffers(self)


class Gemma4ForConditionalGeneration(_Gemma4VLMPrimeRLMixin, HFGemma4ForConditionalGeneration, PreTrainedModelPrimeRL):
    config: Gemma4Config

    def __init__(self, config: Gemma4Config, **kwargs):
        super().__init__(config)
        _mark_gemma4_runtime_buffers_nonpersistent(self)
        final_logit_softcapping = config.get_text_config().final_logit_softcapping
        if final_logit_softcapping is not None:
            config.final_logit_softcapping = final_logit_softcapping

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput | Gemma4CausalLMOutputWithPast:
        assert use_cache is None or use_cache is False, "use_cache is not supported for custom Gemma4"
        assert past_key_values is None, "past_key_values is not supported for custom Gemma4"

        outputs = self._prime_rl_forward_backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )


__all__ = ["Gemma4ForCausalLM", "Gemma4Model", "Gemma4PreTrainedModel", "Gemma4ForConditionalGeneration"]
