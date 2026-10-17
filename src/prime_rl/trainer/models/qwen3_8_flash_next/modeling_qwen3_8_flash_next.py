import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.qwen3_8_flash_next.attention import IndexedGatedAttention
from prime_rl.trainer.models.qwen3_8_flash_next.configuration_qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next.converting_qwen3_8_flash_next import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.qwen3_8_flash_next.gated_delta_net import GatedDeltaNet
from prime_rl.trainer.models.qwen3_8_flash_next.hyper_connection import ExpandedRMSNorm, HyperConnection
from prime_rl.trainer.models.qwen3_8_flash_next.moe import SigmoidOutputGatedMoE
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding import NGramEmbedding
from prime_rl.trainer.models.qwen3_8_flash_next.norm import RMSNorm
from prime_rl.trainer.models.qwen3_8_flash_next.position_learning import PositionLearningEnhancement
from prime_rl.trainer.models.qwen3_8_flash_next.rotary_embedding import RotaryEmbedding, build_mrope_position_ids
from prime_rl.trainer.models.qwen3_8_flash_next.vision import Qwen3_8FlashNextVisionModel
from prime_rl.utils.cp import CPContext, setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class Qwen3_8FlashNextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_8FlashNextTextConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                norm_eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = IndexedGatedAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                norm_eps=config.rms_norm_eps,
                indexer_num_heads=config.indexer_n_heads,
                indexer_head_dim=config.indexer_head_dim,
                indexer_token_budget=config.indexer_budget,
                indexer_compression_ratio=config.indexer_compress_ratio,
            )

        ple_layer_ids = sorted(set(config.ple_layer_ids))
        if layer_index + 1 in ple_layer_ids:
            self.ple = PositionLearningEnhancement(
                hidden_size=config.hidden_size,
                stream_count=config.hc_count,
                embedding_dim=config.ple_embed_dim,
                ngram_size=config.ngram_size,
                heads_per_ngram=config.heads_per_ngram,
                ngram_vocab_size=config.ngram_vocab_size_base,
                token_vocab_size=config.vocab_size,
                eos_token_id=config.eos_token_id,
                vocab_size_divisor=config.make_ngram_vocab_size_divisible_by,
                ngram_layer_index=ple_layer_ids.index(layer_index + 1),
                conv_kernel_size=config.ple_conv_kernel_size,
                norm_eps=config.rms_norm_eps,
            )
        else:
            self.ple = None

        self.mlp = SigmoidOutputGatedMoE(
            dim=config.hidden_size,
            expert_hidden_dim=config.moe_intermediate_size,
            shared_expert_hidden_dim=config.shared_expert_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            activation=config.hidden_act,
            init_std=config.initializer_range,
            load_balance_coeff=config.load_balance_coeff,
        )
        connection_args = {
            "hidden_size": config.hidden_size,
            "stream_count": config.hc_count,
            "low_rank": config.hc_lowrank,
            "norm_eps": config.rms_norm_eps,
        }
        self.attn_hyper_connection = HyperConnection(**connection_args)
        self.mlp_hyper_connection = HyperConnection(**connection_args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        routed_experts: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, cu_seqlens)

        block_input, residual_state = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            block_output = self.linear_attn(block_input, cu_seqlens)
        else:
            block_output = self.self_attn(block_input, position_embeddings, cu_seqlens)
        hidden_states = self.attn_hyper_connection.combine(block_output, residual_state)

        block_input, residual_state = self.mlp_hyper_connection.mix(hidden_states)
        block_output = self.mlp(block_input, routed_experts=routed_experts)
        return self.mlp_hyper_connection.combine(block_output, residual_state)


class Qwen3_8FlashNextPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Qwen3_8FlashNextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_8FlashNextDecoderLayer"]
    _can_compile_fullgraph = False

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(
            frozenset({"ulysses"}),
            "DeltaNet, indexed attention, and PLE require contiguous sequence shards",
        )

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)


class Qwen3_8FlashNextTextModel(Qwen3_8FlashNextPreTrainedModel):
    config_class = Qwen3_8FlashNextTextConfig

    def __init__(self, config: Qwen3_8FlashNextTextConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Qwen3_8FlashNextDecoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.hyper_connection_mixer = HyperConnection(
            hidden_size=config.hidden_size,
            stream_count=config.hc_count,
            low_rank=config.hc_lowrank,
            norm_eps=config.rms_norm_eps,
            with_residual_injection=False,
        )
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            mrope_section=config.mrope_section,
        )
        self.gradient_checkpointing = False
        self.post_init()
        for module in self.modules():
            if isinstance(module, (RMSNorm, ExpandedRMSNorm)):
                module.reset_parameters()
            elif isinstance(module, PositionLearningEnhancement):
                module.reset_parameters()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        """``input_ids`` are always required, image placeholders included: PLE hashes their n-grams even when
        ``inputs_embeds`` carries the image embeddings. ``position_ids`` is ``[batch, seq]`` or ``[3, batch, seq]``
        MRoPE positions."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(
            seq_lens.to(inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds.repeat(1, 1, self.config.hc_count)
        for layer_index, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_index] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                input_ids,
                position_embeddings,
                cu_seqlens,
                routed_experts=layer_routed_experts,
            )
        hidden_states, _ = self.hyper_connection_mixer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3_8FlashNextVLMModel(nn.Module):
    """Composite (``qwen4_exp``) model: image embeddings replace the image placeholder embeddings, while PLE keeps
    hashing the raw placeholder ids."""

    def __init__(self, config: Qwen3_8FlashNextConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_8FlashNextVisionModel(config.vision_config)
        self.language_model = Qwen3_8FlashNextTextModel(config.text_config)
        self.cp_context = CPContext()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(embeddings)

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.LongTensor | None,
        mm_token_type_ids: torch.LongTensor | None,
        seq_lens: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        vision_config = self.config.vision_config
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        has_images = pixel_values is not None
        vision_grid = image_grid_thw
        if has_images:
            pixel_values = pixel_values.to(self.visual.dtype)
        else:
            merge_size = vision_config.spatial_merge_size
            patch_dim = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
            pixel_values = torch.zeros(merge_size**2, patch_dim, device=inputs_embeds.device, dtype=self.visual.dtype)
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
                    raise ValueError("mm_token_type_ids are required with Qwen3.8 Flash Next image inputs")
                position_ids = build_mrope_position_ids(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=vision_config.spatial_merge_size,
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
        if image_grid_thw is not None and self.cp_context.cp_enabled:
            rank, world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
            setup_cp_attention_params(
                position_ids,
                cp_group=self.cp_context.cp_group,
                cp_style=self.cp_context.cp_style,
                seq_lens=seq_lens,
            )
            inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=rank, cp_world_size=world_size)
            input_ids = shard_for_cp(input_ids, cp_rank=rank, cp_world_size=world_size)
            position_ids = shard_position_ids_for_cp(position_ids, cp_rank=rank, cp_world_size=world_size)
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=rank, cp_world_size=world_size)
            seq_lens_are_pre_shard = True

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class Qwen3_8FlashNextForCausalLM(Qwen3_8FlashNextPreTrainedModel):
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3_8FlashNextConfig) -> None:
        super().__init__(config)
        self.is_vlm = hasattr(config, "vision_config")
        text_config = config.text_config if self.is_vlm else config
        self.model = Qwen3_8FlashNextVLMModel(config) if self.is_vlm else Qwen3_8FlashNextTextModel(config)
        self.supports_packed_multimodal_training = self.is_vlm
        self.vocab_size = text_config.vocab_size
        self.num_experts = text_config.num_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        sampling_mask: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
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
            sampling_mask=sampling_mask[:, slice_indices] if sampling_mask is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        language_model = self.model.language_model if self.is_vlm else self.model
        language_model.rotary_emb.reset_parameters()
        if self.is_vlm:
            self.model.visual.rotary_pos_emb.reset_parameters()
        for module in self.modules():
            if isinstance(module, NGramEmbedding):
                module.reset_parameters()
            elif isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()
                if module.router.selection_bias is not None:
                    module.router.selection_bias.zero_()


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextPreTrainedModel",
    "Qwen3_8FlashNextTextModel",
    "Qwen3_8FlashNextVLMModel",
]
