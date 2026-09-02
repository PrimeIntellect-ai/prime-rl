import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)
from prime_rl.trainer.models.qwen3_5.converting_qwen3_5 import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.qwen3_5.gated_delta_net import Qwen3_5GatedDeltaNet
from prime_rl.trainer.models.qwen3_5.norm import Qwen3_5RMSNorm
from prime_rl.trainer.models.qwen3_5.rotary_embedding import (
    Qwen3_5RotaryEmbedding,
    build_qwen3_5_mrope_position_ids,
)
from prime_rl.trainer.models.qwen3_5.vision import Qwen3_5VisionModel
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class Qwen3_5SharedExpert(FeedForward):
    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__(
            dim=config.hidden_size,
            hidden_dim=config.shared_expert_intermediate_size,
            expert_type="gated",
            activation=config.hidden_act,
        )
        self.output_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        output = super().forward(hidden_states, routed_experts)
        return output * self.output_gate(hidden_states).sigmoid()


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, config._attn_implementation)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        if isinstance(config, Qwen3_5MoeTextConfig):
            router = TokenChoiceTopKRouter(
                dim=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
            )
            experts = GroupedExperts(
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.num_experts,
                expert_type="gated",
                activation=config.hidden_act,
            )
            experts.init_weights(config.initializer_range)
            self.mlp = MoE(
                router=router,
                experts=experts,
                shared_expert=Qwen3_5SharedExpert(config),
                score_before_experts=False,
                load_balance_coeff=config.load_balance_coeff,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                expert_type="gated",
                activation=config.hidden_act,
            )

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
        routed_experts: torch.LongTensor | None = None,
        *,
        cu_seqlens_are_pre_shard: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states,
                cu_seqlens,
                cu_seqlens_are_pre_shard=cu_seqlens_are_pre_shard,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                cu_seqlens,
                max_seqlen,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states, routed_experts=routed_experts)


class Qwen3_5PreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)


class Qwen3_5Model(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()
        # Qwen stores RMSNorm weights as offsets from one, while post_init treats them as direct scales.
        for module in self.modules():
            if isinstance(module, Qwen3_5RMSNorm):
                module.reset_parameters()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def set_context_parallel_attributes(self, process_group, rank: int, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.context_parallel_rank = rank
        self.context_parallel_world_size = world_size
        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.set_context_parallel_attributes(process_group, world_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"Qwen3.5 expects one packed row, got batch size {inputs_embeds.shape[0]}")
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer_index, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_index] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                cu_seqlens,
                max_seqlen,
                routed_experts=layer_routed_experts,
                cu_seqlens_are_pre_shard=seq_lens_are_pre_shard,
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


class Qwen3_5VLMModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5Model(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(embeddings)

    def set_context_parallel_attributes(self, process_group, rank: int, world_size: int) -> None:
        self.language_model.set_context_parallel_attributes(process_group, rank, world_size)

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.LongTensor | None,
        mm_token_type_ids: torch.LongTensor | None,
        seq_lens: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        has_images = pixel_values is not None
        vision_grid = image_grid_thw
        if has_images:
            pixel_values = pixel_values.to(self.visual.dtype)
        else:
            merge_size = self.config.vision_config.spatial_merge_size
            num_patches = merge_size**2
            patch_dim = (
                self.config.vision_config.in_channels
                * self.config.vision_config.temporal_patch_size
                * self.config.vision_config.patch_size**2
            )
            pixel_values = torch.zeros(
                num_patches,
                patch_dim,
                device=inputs_embeds.device,
                dtype=self.visual.dtype,
            )
            vision_grid = torch.tensor([[1, merge_size, merge_size]], device=inputs_embeds.device)

        image_embeds = self.visual(pixel_values, vision_grid).pooler_output.to(inputs_embeds.dtype)
        if has_images:
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            # Every rank must retain the vision graph so FSDP collectives stay symmetric.
            inputs_embeds = inputs_embeds + image_embeds.sum() * 0.0

        if position_ids is None:
            if image_grid_thw is None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                if mm_token_type_ids is None:
                    raise ValueError("mm_token_type_ids are required with Qwen3.5 image inputs")
                position_ids = build_qwen3_5_mrope_position_ids(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    seq_lens=seq_lens,
                )
        return inputs_embeds, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        inputs_embeds, position_ids = self.prepare_inputs(
            input_ids,
            position_ids,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            seq_lens,
        )
        process_group = getattr(self.language_model, "context_parallel_group", None)
        if image_grid_thw is not None and process_group is not None:
            rank = self.language_model.context_parallel_rank
            world_size = self.language_model.context_parallel_world_size
            setup_cp_attention_params(position_ids, cp_group=process_group, cp_style="ulysses", seq_lens=seq_lens)
            inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=rank, cp_world_size=world_size)
            position_ids = shard_position_ids_for_cp(position_ids, cp_rank=rank, cp_world_size=world_size)
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=rank, cp_world_size=world_size)
            seq_lens_are_pre_shard = True

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.is_vlm = hasattr(config, "vision_config")
        text_config = config.text_config if self.is_vlm else config
        attention_implementation = (
            getattr(config, "_attn_implementation", None)
            or getattr(text_config, "_attn_implementation", None)
            or "flash_attention_3"
        )
        text_config._attn_implementation = attention_implementation

        if self.is_vlm:
            if getattr(config.vision_config, "_attn_implementation_internal", None) is None:
                config.vision_config._attn_implementation = attention_implementation
            self.model = Qwen3_5VLMModel(config)
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        else:
            self.model = Qwen3_5Model(config)

        self.supports_packed_multimodal_training = self.is_vlm
        self.vocab_size = text_config.vocab_size
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)
        if isinstance(text_config, Qwen3_5MoeTextConfig):
            self.num_experts = text_config.num_experts
            self.num_experts_per_tok = text_config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def set_context_parallel_attributes(self, process_group, rank: int, world_size: int) -> None:
        self.model.set_context_parallel_attributes(process_group, rank, world_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        if kwargs.get("use_cache") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("Qwen3.5 custom training does not support KV caching")
        if self.is_vlm:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            )

        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        return self.lm_head(
            outputs.last_hidden_state[:, slice_indices],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        language_model = self.model.language_model if self.is_vlm else self.model
        language_model.rotary_emb.reset_parameters()
        if self.is_vlm:
            self.model.visual.rotary_pos_emb.reset_parameters()
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()
                if module.router.selection_bias is not None:
                    module.router.selection_bias.zero_()


__all__ = [
    "Qwen3_5Attention",
    "Qwen3_5DecoderLayer",
    "Qwen3_5ForCausalLM",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
]
