import warnings
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div, fp8_block_quantize, grouped_fp8_block_quantize
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    _BASE,
    _DENSE,
    _SPARSE,
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_layer_to_vllm_kernel,
    convert_tt_to_hf_moe,
)
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.cp import gather_for_cp, shard_for_cp


def _sparse_mla_attention_args(config: GlmMoeDsaConfig) -> SparseMlaAttentionArgs:
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
    )


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(_sparse_mla_attention_args(config))

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

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return hasattr(self, "_cp_group") and hasattr(self, "_cp_rank") and hasattr(self, "_cp_world_size")

    def shard_to_cp(self, t: torch.Tensor) -> torch.Tensor:
        if not self.cp_enabled:
            return t

        return shard_for_cp(t, self._cp_rank, self._cp_world_size)

    def gather_for_cp(self, t: torch.Tensor) -> torch.Tensor:
        if not self.cp_enabled:
            return t

        return gather_for_cp(t, self._cp_group)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ks: Optional[torch.Tensor] = None,
        ke: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.gather_for_cp(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
        )
        hidden_states = self.shard_to_cp(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
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

    def convert_layer_to_vllm_kernel(self, layer_idx: int, out_buffers: dict[str, Tensor]) -> None:
        """Quantize/copy one layer's tensors into ``out_buffers``.

        Expert specs — fused cat-then-quantize into an EP+FSDP-local slot.

        Non-expert specs — one slot per source. The slot's dim-0 size tells us
        the handling:
          - equal to the source's full dim 0 → ``gather``: quantize the full
            tensor (via ``full_tensor()`` for DTensor, as-is for plain).
          - smaller → ``per_shard``: take this rank's ``dim0/fsdp_size`` slab,
            either via ``to_local()`` on the DTensor or by slicing the plain
            tensor (test path — the assumption that plain tensors carry the
            full content matches how the test fixtures are set up).
        """
        prefix = f"model.layers.{layer_idx}"
        state_dict = self.state_dict()
        is_sparse = f"{prefix}.mlp.router.gate.weight" in state_dict
        specs = _BASE + (_SPARSE if is_sparse else _DENSE)

        my_rank = dist.get_rank() if dist.is_initialized() else 0

        for spec in specs:
            is_expert = spec.dst.startswith("mlp.experts.")
            if is_expert:
                srcs: list[Tensor] = []
                for name in spec.sources:
                    s = state_dict[f"{prefix}.{name}"]
                    s = s.to_local() if isinstance(s, DTensor) else s
                    srcs.append(s.to(torch.bfloat16))
                tensor = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=spec.cat_dim)
                dst = f"{prefix}.{spec.dst}"
                if spec.quantize:
                    scale_name = spec.scale_name(prefix)
                    quantize = grouped_fp8_block_quantize if tensor.ndim == 3 else fp8_block_quantize
                    quantize(tensor, out=out_buffers[dst], sf=out_buffers[scale_name])
                else:
                    out_buffers[dst].copy_(tensor)
                continue

            for src_name in spec.sources:
                src = state_dict[f"{prefix}.{src_name}"]
                slot_key = f"{prefix}.{src_name}"
                slot = out_buffers[slot_key]
                src_rows_full = src.shape[0]  # DTensor.shape is global; plain is its actual rows
                if slot.shape[0] == src_rows_full:
                    value = src.full_tensor() if isinstance(src, DTensor) else src
                else:
                    if isinstance(src, DTensor):
                        value = src.to_local()
                    else:
                        step = slot.shape[0]
                        value = src[my_rank * step : (my_rank + 1) * step].contiguous()
                value = value.to(torch.bfloat16)
                if spec.quantize:
                    scale_key = slot_key.removesuffix(".weight") + ".weight_scale_inv"
                    fp8_block_quantize(value, out=slot, sf=out_buffers[scale_key])
                else:
                    slot.copy_(value)

    def allocate_slots(self, parallel_dims: ParallelDims) -> dict[int, dict[str, Tensor]]:
        """Stable destination buffers for NIXL weight transfer.

        Experts: one fused slot per spec, EP+FSDP-local shape.
        Non-experts: one slot per source, shape determined by handling:
          - per-shard when ``src_rows % fsdp_total == 0`` AND (for quantized specs)
            ``(src_rows / fsdp_total) % BLOCK_SIZE == 0`` — slot holds the
            rank-local FSDP shard.
          - gather otherwise — slot holds the full tensor.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        bf16 = torch.bfloat16
        fsdp_total = parallel_dims.dp_shard * parallel_dims.cp

        slots: dict[int, dict[str, Tensor]] = {}
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            is_sparse = f"{prefix}.mlp.router.gate.weight" in state_dict
            specs = _BASE + (_SPARSE if is_sparse else _DENSE)

            layer_slots: dict[str, Tensor] = {}
            for spec in specs:
                if spec.dst.startswith("mlp.experts."):
                    src_local_shapes = [
                        (s.to_local() if isinstance(s, DTensor) else s).shape
                        for s in (state_dict[f"{prefix}.{name}"] for name in spec.sources)
                    ]
                    dst_shape = list(src_local_shapes[0])
                    dst_shape[spec.cat_dim] = sum(sh[spec.cat_dim] for sh in src_local_shapes)
                    dst_dtype = torch.float8_e4m3fn if spec.quantize else bf16
                    layer_slots[f"{prefix}.{spec.dst}"] = torch.empty(dst_shape, dtype=dst_dtype, device=device)
                    if spec.quantize:
                        scale_shape = tuple(
                            ceil_div(d, BLOCK_SIZE) if i >= len(dst_shape) - 2 else d
                            for i, d in enumerate(dst_shape)
                        )
                        layer_slots[spec.scale_name(prefix)] = torch.empty(
                            scale_shape, dtype=torch.float32, device=device
                        )
                    continue

                for src_name in spec.sources:
                    src = state_dict[f"{prefix}.{src_name}"]
                    full_shape = tuple(src.shape)
                    src_rows = full_shape[0]
                    per_shard = src_rows % fsdp_total == 0 and (
                        not spec.quantize or (src_rows // fsdp_total) % BLOCK_SIZE == 0
                    )
                    rows = src_rows // fsdp_total if per_shard else src_rows
                    slot_shape = (rows,) + full_shape[1:]
                    slot_dtype = torch.float8_e4m3fn if spec.quantize else bf16
                    slot_key = f"{prefix}.{src_name}"
                    layer_slots[slot_key] = torch.empty(slot_shape, dtype=slot_dtype, device=device)
                    if spec.quantize:
                        scale_shape = tuple(
                            ceil_div(d, BLOCK_SIZE) if i >= len(slot_shape) - 2 else d
                            for i, d in enumerate(slot_shape)
                        )
                        scale_key = slot_key.removesuffix(".weight") + ".weight_scale_inv"
                        layer_slots[scale_key] = torch.empty(scale_shape, dtype=torch.float32, device=device)
            slots[layer_idx] = layer_slots
        return slots

    def non_expert_slot_layout(self, parallel_dims: ParallelDims) -> dict[int, dict[str, dict]]:
        """Per non-expert slot key → inference-side placement info.

        Entries:
          - ``inference_name``: vLLM tensor name (fused dst param / scale buffer).
          - ``offset_rows``: row offset of this slot within ``inference_name``.
          - ``handling``: ``"per_shard"`` or ``"gather"``.
        """
        state_dict = self.state_dict()
        fsdp_total = parallel_dims.dp_shard * parallel_dims.cp
        layout: dict[int, dict[str, dict]] = {}
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            is_sparse = f"{prefix}.mlp.router.gate.weight" in state_dict
            specs = _BASE + (_SPARSE if is_sparse else _DENSE)
            layer_layout: dict[str, dict] = {}

            for spec in specs:
                if spec.dst.startswith("mlp.experts."):
                    continue
                inference_value = f"{prefix}.{spec.dst}"
                inference_scale = spec.scale_name(prefix) if spec.quantize else ""
                row_off = 0
                scale_row_off = 0
                for src_name in spec.sources:
                    src = state_dict[f"{prefix}.{src_name}"]
                    src_rows = src.shape[0]
                    per_shard = src_rows % fsdp_total == 0 and (
                        not spec.quantize or (src_rows // fsdp_total) % BLOCK_SIZE == 0
                    )
                    handling = "per_shard" if per_shard else "gather"
                    slot_key = f"{prefix}.{src_name}"
                    layer_layout[slot_key] = {
                        "inference_name": inference_value,
                        "offset_rows": row_off,
                        "rows": src_rows,
                        "handling": handling,
                    }
                    if spec.quantize:
                        scale_key = slot_key.removesuffix(".weight") + ".weight_scale_inv"
                        scale_rows = ceil_div(src_rows, BLOCK_SIZE)
                        layer_layout[scale_key] = {
                            "inference_name": inference_scale,
                            "offset_rows": scale_row_off,
                            "rows": scale_rows,
                            "handling": handling,
                        }
                        scale_row_off += scale_rows
                    row_off += src_rows
            layout[layer_idx] = layer_layout
        return layout


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

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        """
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        cp_group, _, cp_world_size = self._context_parallel_state()
        if cp_group is not None and cp_world_size > 1:
            position_ids_for_attn = self._gather_position_ids_for_cp(position_ids, cp_group, cp_world_size)
        else:
            position_ids_for_attn = position_ids

        flat_position_ids = position_ids_for_attn.view(-1)
        S = flat_position_ids.shape[0]
        ks = torch.arange(S, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(torch.int32)
        ke = torch.arange(1, S + 1, dtype=torch.int32, device=flat_position_ids.device)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids_for_attn)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
                routed_experts=routed_experts_layer,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
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
        routed_experts: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
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
            routed_experts=routed_experts,
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
