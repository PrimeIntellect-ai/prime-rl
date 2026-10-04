import os

import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.base import ALL_CP_STYLES, CPSupport, PreTrainedModelPrimeRL
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
from prime_rl.utils.cp import (
    CPContext,
    gather_for_cp,
    setup_cp_attention_params,
    shard_for_cp,
    shard_position_ids_for_cp,
)
from prime_rl.utils.logger import get_logger
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

        if isinstance(config, Qwen3_5MoeTextConfig):
            router = TokenChoiceTopKRouter(
                dim=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                selection_bias=config.load_balance_coeff is not None,
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states,
                cu_seqlens,
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

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        # VLM configs nest the layer schedule under `text_config`.
        text_config = getattr(config, "text_config", config)
        if "linear_attention" in (getattr(text_config, "layer_types", None) or ()):
            return CPSupport(
                frozenset({"ulysses"}),
                "DeltaNet layers require Ulysses sequence sharding with FLA boundary-state exchange",
            )
        return CPSupport(ALL_CP_STYLES)

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
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

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
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
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
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


# ---------------------------------------------------------------------------
# Per-CP-rank vision encoding (opt-in). Default behaviour is unchanged: every CP
# rank encodes every image. Set PRIME_RL_VLM_CP_PER_RANK_ENCODE=1 to partition
# images across CP ranks and all-gather the embeddings instead. Set
# PRIME_RL_VLM_TIME_VISION=1 to log vision-encoder wall time on rank 0.
# ---------------------------------------------------------------------------
_PER_RANK_ENCODE = os.environ.get("PRIME_RL_VLM_CP_PER_RANK_ENCODE", "0") == "1"
_TIME_VISION = os.environ.get("PRIME_RL_VLM_TIME_VISION", "0") == "1"
_TIME_VISION_EVERY = int(os.environ.get("PRIME_RL_VLM_TIME_VISION_EVERY", "20"))


def _partition_images_for_cp(image_grid_thw: torch.LongTensor, cp_world_size: int) -> list[list[int]]:
    """Greedy largest-first balance of images across CP ranks by patch count. Deterministic, no comm."""
    patches = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
    order = sorted(range(len(patches)), key=lambda i: -patches[i])
    buckets: list[list[int]] = [[] for _ in range(cp_world_size)]
    load = [0] * cp_world_size
    for i in order:
        r = min(range(cp_world_size), key=lambda r: load[r])
        buckets[r].append(i)
        load[r] += patches[i]
    return buckets


@torch.compiler.disable
def _encode_images_per_cp_rank(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    merge_size: int,
    cp_context: CPContext,
    requires_grad: bool,
) -> torch.Tensor:
    """Encode this rank's share of the images, all-gather, return embeddings in original image order.

    Invariants: `visual` is called exactly once on every rank (FSDP symmetry); the partition is
    computed identically on every rank from `image_grid_thw`; output rows follow image order so the
    caller's masked_scatter is unchanged.
    """
    device = pixel_values.device
    rank, world = cp_context.cp_rank, cp_context.cp_world_size
    patches = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    tokens = patches // (merge_size**2)
    patch_off = torch.cumsum(patches, 0) - patches
    buckets = _partition_images_for_cp(image_grid_thw, world)
    mine = buckets[rank]

    if mine:
        rows = torch.cat([torch.arange(int(patch_off[i]), int(patch_off[i] + patches[i]), device=device) for i in mine])
        local = visual(pixel_values[rows], image_grid_thw[mine]).pooler_output
    else:
        dummy = torch.zeros(merge_size**2, pixel_values.shape[1], device=device, dtype=pixel_values.dtype)
        dummy_grid = torch.tensor([[1, merge_size, merge_size]], device=device)
        local = visual(dummy, dummy_grid).pooler_output[:0]

    hidden = local.shape[-1]
    per_rank_tokens = [int(tokens[b].sum()) if b else 0 for b in buckets]
    max_tokens = max(per_rank_tokens)
    padded = local.new_zeros(max_tokens, hidden)
    if local.shape[0]:
        padded[: local.shape[0]] = local

    if requires_grad:
        gathered = gather_for_cp(padded.unsqueeze(0), cp_context.cp_group).squeeze(0)
        chunks = list(gathered.split(max_tokens, dim=0))
    else:
        chunks = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(chunks, padded.contiguous(), group=cp_context.cp_group)

    per_image: list[torch.Tensor | None] = [None] * int(image_grid_thw.shape[0])
    for r, b in enumerate(buckets):
        off = 0
        for i in b:
            n = int(tokens[i])
            per_image[i] = chunks[r][off : off + n]
            off += n
    return torch.cat(per_image, dim=0)


class _VisionTimer:
    """Accumulates CUDA-event wall time of the vision encoder; logs on rank 0 every N forwards."""

    def __init__(self) -> None:
        self.calls = 0
        self.total_s = 0.0
        self.window_s = 0.0

    def __call__(self, fn, *args, **kwargs):
        if not _TIME_VISION or not torch.cuda.is_available():
            return fn(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        end.synchronize()
        dt = start.elapsed_time(end) / 1000.0
        self.calls += 1
        self.total_s += dt
        self.window_s += dt
        if self.calls % _TIME_VISION_EVERY == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            get_logger().info(
                f"[vision-timer] last {_TIME_VISION_EVERY} fwd: {self.window_s:.3f}s "
                f"({self.window_s / _TIME_VISION_EVERY * 1000:.1f} ms/fwd) | cumulative {self.total_s:.1f}s "
                f"| per_rank_encode={_PER_RANK_ENCODE}"
            )
            self.window_s = 0.0
        return out


_vision_timer = _VisionTimer()


class Qwen3_5VLMModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5Model(config.text_config)
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

        use_per_rank = (
            _PER_RANK_ENCODE and has_images and self.cp_context.cp_enabled and self.cp_context.cp_world_size > 1
        )
        if use_per_rank:
            image_embeds = _vision_timer(
                _encode_images_per_cp_rank,
                self.visual,
                pixel_values,
                vision_grid,
                self.config.vision_config.spatial_merge_size,
                self.cp_context,
                any(p.requires_grad for p in self.visual.parameters()),
            ).to(inputs_embeds.dtype)
        else:
            image_embeds = _vision_timer(self.visual, pixel_values, vision_grid).pooler_output.to(inputs_embeds.dtype)
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
        if image_grid_thw is not None and self.cp_context.cp_enabled:
            rank, world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
            setup_cp_attention_params(
                position_ids,
                cp_group=self.cp_context.cp_group,
                cp_style=self.cp_context.cp_style,
                seq_lens=seq_lens,
            )
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
    def __init__(self, config) -> None:
        super().__init__(config)
        self.is_vlm = hasattr(config, "vision_config")
        text_config = config.text_config if self.is_vlm else config

        if self.is_vlm:
            self.model = Qwen3_5VLMModel(config)
        else:
            self.model = Qwen3_5Model(config)

        self.supports_packed_multimodal_training = self.is_vlm
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        temperature: torch.Tensor | None = None,
        sampling_mask: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
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
                inputs_embeds=inputs_embeds,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            )

        return self.lm_head(
            outputs.last_hidden_state,
            labels,
            temperature=temperature,
            sampling_mask=sampling_mask,
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
