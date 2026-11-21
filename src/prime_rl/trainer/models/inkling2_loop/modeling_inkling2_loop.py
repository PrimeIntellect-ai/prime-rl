"""Looped Inkling 2 decoder (Inkling 2 Mini), text-only.

Reference implementations:
  * transformers ``Inkling2LoopForCausalLM`` (huggingface/transformers#51050, read at commit 6173a2f)
  * the Megatron reference written for SkyRL (``reference.py`` in the folder that ``dequantize.py`` links)

Two things set the decoder apart from a plain residual stack:

- There is no single residual stream. Layers come in blocks of ``n_layers_per_attn_res_block``. Inside a block
  the layer outputs add up into a running sum, and at a block boundary that sum is put aside as a block state
  and the sum restarts. Every sub-layer reads a learned mix (``DepthMix``) of the block states and the running
  sum instead of the sum itself; only layer 0's attention reads the embedding directly.
- The loop core runs ``num_passes`` times with shared weights. Every pass restarts from the state the layers
  before the core left. Passes after the first also read messages, mixes of the previous pass's block outputs,
  which ``MessageGate`` blends into each sub-layer input.

The model takes one packed row of documents. Documents are isolated in attention and in every convolution, and
positions restart at each document.

Context parallelism splits the row into equal shards along the sequence, one per rank. Everything is per token
except attention, which the ranks compute head-sharded over the whole row, and the convolutions, which read the few
tokens before a shard from the rank that holds them.
"""

from functools import cache

import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL, TextModelOutput, count_active_params
from prime_rl.trainer.models.inkling2_loop.attention import Inkling2Attention
from prime_rl.trainer.models.inkling2_loop.configuration_inkling2_loop import (
    SLIDING_ATTENTION,
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
)
from prime_rl.trainer.models.inkling2_loop.converting_inkling2_loop import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.inkling2_loop.dequantize import dequantize_state_dict_
from prime_rl.trainer.models.inkling2_loop.layers import (
    DepthMix,
    DynamicShortConv,
    Inkling2FinalNorm,
    Inkling2RMSNorm,
    MessageGate,
    StaticShortConv,
    block_sources,
    depth_mix,
)
from prime_rl.trainer.models.layers.activations import SoftcappedSwiglu
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, TokenSupport, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

MESSAGE_STREAMS = ("q", "k", "v", "mlp")
# The checkpoint stores these tensors in FP32, and they go on the wire in FP32. The trainer's forward computes
# with the BF16 rounding of those that are parameters: FSDP casts every parameter to its compute dtype.
FP32_CHECKPOINT_TENSORS = (
    "attention_input.query",
    "mlp_input.query",
    "norm.query",
    "message_queries",
    "selection_bias",
)


@cache
def softcapped_swiglu(gate_softcap: float, up_softcap: float) -> type[SoftcappedSwiglu]:
    """``SoftcappedSwiglu`` with these caps. The shared class holds the released caps as class attributes, which its
    ``apply`` reads, so other caps (a tiny test config's) go into a subclass: set on an instance, they would be
    ignored. One subclass per pair of caps: a compiled layer is guarded on the identity of its activation, and layers
    with different ones would not share a graph. Such a subclass is made here and cannot be pickled by name."""
    if (gate_softcap, up_softcap) == (SoftcappedSwiglu.gate_softcap, SoftcappedSwiglu.up_softcap):
        return SoftcappedSwiglu
    namespace = {
        "gate_softcap": gate_softcap,
        "up_softcap": up_softcap,
        "__module__": __name__,
        "__qualname__": f"SoftcappedSwiglu[{gate_softcap}, {up_softcap}]",
    }
    return type("SoftcappedSwiglu", (SoftcappedSwiglu,), namespace)


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        in_loop_core = layer_idx in config.loop_core
        hidden_size, eps = config.hidden_size, config.rms_norm_eps

        # Full layers read separate mixes for their query, key and value inputs. Layer 0 reads the embedding.
        self.attention_input = DepthMix(hidden_size, 1 if self.is_sliding else 3) if layer_idx > 0 else None
        self.mlp_input = DepthMix(hidden_size)
        if self.is_sliding:
            self.input_layernorm = Inkling2RMSNorm(hidden_size, eps)
        else:
            self.q_input_layernorm = Inkling2RMSNorm(hidden_size, eps)
            self.k_input_layernorm = Inkling2RMSNorm(hidden_size, eps)
            self.v_input_layernorm = Inkling2RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = Inkling2RMSNorm(hidden_size, eps)
        self.self_attn = Inkling2Attention(config, layer_idx, num_visits=config.loop_num_passes if in_loop_core else 1)
        self.attn_conv = StaticShortConv(hidden_size, config.sconv_kernel_size)
        self.mlp_conv = StaticShortConv(hidden_size, config.sconv_kernel_size)
        self.mlp = MoE(
            router=TokenChoiceTopKRouter(
                dim=hidden_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                score_func="sigmoid",
                route_norm=True,
                route_scale=config.route_scale,
                # The checkpoint's `gate.bias` shifts which experts are selected, not their weights.
                selection_bias=True,
            ),
            experts=GroupedExperts(
                dim=hidden_size,
                hidden_dim=config.intermediate_size,
                num_experts=config.n_routed_experts,
                expert_type="gated",
                activation=softcapped_swiglu(config.mlp_gate_softcap, config.mlp_up_softcap),
            ),
            shared_expert=None,
            score_before_experts=False,
            load_balance_coeff=None,
        )
        self.message_gates = (
            nn.ModuleDict({stream: MessageGate(hidden_size, config.loop_gate_rank, eps) for stream in MESSAGE_STREAMS})
            if in_loop_core
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        positions: torch.Tensor,
        first_key: torch.Tensor,
        message_bank: torch.Tensor | None = None,
        message_queries: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[1, T, H]``, the running sum of the current block.
            block_states: ``[1, T, slots, H]``, one slot per block of the model.
            block_mask: ``[slots]``, the slots that hold a finished block.
            positions: ``[T]``, token index inside its document.
            first_key: ``[row]``, the first key each token of the row attends (see ``rel_attention``). The row is
                this rank's ``T`` tokens, or under context parallelism every rank's.
            message_bank: ``[1, T, m, H]``, the previous pass's block outputs. Only in a later pass of the loop.
            message_queries: ``[4, H]``, this layer's message queries for its query, key, value and MLP inputs.
            routed_experts: ``[1, T, top_k]``, the experts to route this visit to instead of the router's picks.

        Returns:
            The next running sum ``[1, T, H]``. At a block boundary ``hidden_states`` becomes a finished block,
            which the caller stores in the next slot.
        """
        dtype = hidden_states.dtype
        if self.attention_input is None:
            streams = (hidden_states,)
        else:
            streams = self.attention_input(*block_sources(block_states, block_mask, hidden_states)).unbind(2)

        messages = None
        if message_bank is not None:
            messages = depth_mix(message_bank, message_queries).unbind(2)
            streams = tuple(
                self.message_gates[name](stream, message, dtype)
                for name, stream, message in zip("qkv", streams * (3 // len(streams)), messages)
            )

        if self.is_sliding:
            normed = tuple(self.input_layernorm(stream, dtype) for stream in streams)
        else:
            norms = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
            normed = tuple(norm(stream, dtype) for norm, stream in zip(norms, streams * (3 // len(streams))))
        query_states, key_states, value_states = normed * (3 // len(normed))
        attn_output = self.self_attn(query_states, key_states, value_states, positions, first_key)
        attn_output = self.attn_conv(attn_output, None, positions)

        if self.is_block_boundary:
            residual = attn_output
            mlp_sources = block_sources(block_states, block_mask, hidden_states, residual)
        else:
            residual = hidden_states.float() + attn_output.float()
            mlp_sources = block_sources(block_states, block_mask, residual)
        mlp_states = self.mlp_input(*mlp_sources)[:, :, 0]
        if messages is not None:
            mlp_states = self.message_gates["mlp"](mlp_states, messages[3], dtype)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_states, dtype), routed_experts=routed_experts)
        mlp_output = self.mlp_conv(mlp_output, None, positions)
        return (residual.float() + mlp_output.float()).to(dtype)


class Inkling2LoopPreTrainedModel(PreTrainedModelPrimeRL):
    config: Inkling2LoopConfig | Inkling2LoopTextConfig

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        # Attention is head-sharded without KV-head replication, on the full and the sliding layers.
        text_config = getattr(config, "text_config", config)
        head_counts = {
            name: getattr(text_config, name)
            for name in (
                "num_attention_heads",
                "num_key_value_heads",
                "swa_num_attention_heads",
                "swa_num_key_value_heads",
            )
        }
        indivisible = [f"{name}={count}" for name, count in head_counts.items() if count % cp != 0]
        if indivisible:
            return f"cp={cp} must divide every attention head count, which it does not for {', '.join(indivisible)}"
        return None

    @classmethod
    def lora_blocker(cls, config) -> str | None:
        return (
            "its adapters have no export to the serving stack's layout (checkpoint names, one stacked adapter per "
            "expert projection), and the loop message gates take no adapter there"
        )

    @classmethod
    def cuda_graph_blocker(cls, config) -> str | None:
        return (
            "the attention kernel plans its query chunks from the row's document layout on the host, which a CUDA "
            "graph cannot record"
        )

    @classmethod
    def dynamic_layer_arg_dims(cls) -> tuple[tuple[str, int], ...]:
        # Every one of them: an arg left static pins the row length it shares with the others.
        return (
            ("hidden_states", 1),
            ("block_states", 1),
            ("positions", 0),
            ("first_key", 0),
            ("message_bank", 1),
            ("routed_experts", 1),
        )

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(tensor_name in name for tensor_name in FP32_CHECKPOINT_TENSORS)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a checkpoint state dict to PrimeRL format in-place, dequantizing a release's packed experts
        first (a no-op for a BF16 export)."""
        dequantize_state_dict_(state_dict)
        return super().convert_to_prime(state_dict)

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, Inkling2Attention):
            nn.init.ones_(module.q_scale)
            nn.init.ones_(module.rel_scale)
            nn.init.normal_(module.rel_table, mean=0.0, std=std)
        elif isinstance(module, StaticShortConv):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DynamicShortConv):
            nn.init.zeros_(module.static_weight)
        elif isinstance(module, DepthMix):
            nn.init.zeros_(module.query)
        elif isinstance(module, Inkling2RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Inkling2FinalNorm):
            nn.init.zeros_(module.query)
            nn.init.ones_(module.weight)
        elif isinstance(module, Inkling2LoopModel) and module.message_queries is not None:
            nn.init.zeros_(module.message_queries)

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()


class Inkling2LoopModel(Inkling2LoopPreTrainedModel):
    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__(config)
        unsupported = config.unsupported_variant()
        if unsupported is not None:
            raise NotImplementedError(f"PrimeRL cannot train this Inkling 2 checkpoint: {unsupported}")
        self.sliding_window = config.sliding_window_size
        self.loop_core = config.loop_core
        self.loop_num_passes = config.loop_num_passes
        self.num_layer_visits = config.num_hidden_layers + (self.loop_num_passes - 1) * len(self.loop_core)
        self.num_blocks = -(-config.num_hidden_layers // config.n_layers_per_attn_res_block)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        # Per core layer of the checkpoint, the message queries of its query, key, value and MLP inputs. A model
        # truncated inside the core uses the rows of the core layers it loops.
        checkpoint_core_layers = config.loop_config["core_end"] - config.loop_config["core_start"]
        self.message_queries = (
            nn.Parameter(torch.zeros(checkpoint_core_layers, len(MESSAGE_STREAMS), config.hidden_size))
            if len(self.loop_core) > 0
            else None
        )
        self.norm = Inkling2FinalNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def layer_visit_ids(self, layer_idx: int) -> list[int]:
        """The visits of a layer in execution order (the layers before the core, every pass of the core, the layers
        after it): one per pass for a core layer, else one. They index the layer axis of ``routed_experts``."""
        core, extra = self.loop_core, (self.loop_num_passes - 1) * len(self.loop_core)
        if layer_idx < core.start:
            return [layer_idx]
        if layer_idx >= core.stop:
            return [layer_idx + extra]
        return [layer_idx + pass_idx * len(core) for pass_idx in range(self.loop_num_passes)]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> TextModelOutput:
        """
        position_ids: Ignored. Positions are derived from ``seq_lens``.
        routed_experts (`torch.LongTensor` of shape `(1, sequence_length, num_layer_visits, num_experts_per_tok)`):
            Routed experts of every layer visit (see ``layer_visit_ids``). Only used for router replay.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row.
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether the other inputs are this rank's context-parallel shard of the row that `seq_lens` describes.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        cp = self.cp_context
        if seq_lens_are_pre_shard != cp.cp_enabled:
            raise ValueError(
                f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} does not match context parallelism "
                f"(cp={cp.cp_world_size}): the inputs are a shard of the row exactly when it is on"
            )
        if routed_experts is not None and routed_experts.shape[2] != self.num_layer_visits:
            raise ValueError(
                f"routed_experts holds {routed_experts.shape[2]} layers, but the decoder visits "
                f"{self.num_layer_visits} (the loop core counts once per pass)"
            )

        if inputs_embeds is None:
            embeddings = self.embed_tokens(input_ids)
            inputs_embeds = self.embed_norm(embeddings, embeddings.dtype)
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"Inkling 2 takes one packed row, got batch size {inputs_embeds.shape[0]}")

        num_tokens = inputs_embeds.shape[1]
        if cp.cp_enabled and num_tokens < self.config.sconv_kernel_size - 1:
            raise ValueError(
                f"A context-parallel shard of {num_tokens} tokens is shorter than the "
                f"{self.config.sconv_kernel_size - 1} tokens the convolutions of the next shard read"
            )
        row_tokens = num_tokens * cp.cp_world_size
        device = inputs_embeds.device
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens.to(device=device), total_tokens=row_tokens)
        token = torch.arange(row_tokens, device=device)
        document_start = torch.repeat_interleave(
            cu_seqlens[:-1].long(), seq_lens.to(device=device, dtype=torch.long), output_size=row_tokens
        )
        # Attention sees the whole row. Everything else sees this rank's tokens, as a copy: a compiled layer's
        # input must not be a view of a larger tensor.
        shard = slice(cp.cp_rank * num_tokens, (cp.cp_rank + 1) * num_tokens)
        positions = (token - document_start)[shard].clone()
        first_key_full = document_start
        first_key_sliding = torch.maximum(document_start, token - (self.sliding_window - 1))

        num_visits = 0
        block_slots = torch.arange(self.num_blocks, device=device)

        def run(layer, hidden_states, block_states, num_finished, message_bank=None, message_queries=None):
            nonlocal num_visits
            # By keyword: `dynamic_layer_arg_dims` names the args whose row length torch.compile leaves dynamic.
            output = layer(
                hidden_states=hidden_states,
                block_states=block_states,
                block_mask=block_slots < num_finished,
                positions=positions,
                first_key=first_key_sliding if layer.is_sliding else first_key_full,
                message_bank=message_bank,
                message_queries=message_queries,
                routed_experts=routed_experts[:, :, num_visits].contiguous() if routed_experts is not None else None,
            )
            num_visits += 1
            if layer.is_block_boundary:
                block_states = torch.cat(
                    [
                        block_states[:, :, :num_finished],
                        hidden_states.unsqueeze(2),
                        block_states[:, :, num_finished + 1 :],
                    ],
                    dim=2,
                )
                num_finished += 1
            return output, block_states, num_finished

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros((1, num_tokens, self.num_blocks, inputs_embeds.shape[-1]))
        num_finished = 0
        core = self.loop_core
        for layer in self.layers[: core.start]:
            hidden_states, block_states, num_finished = run(layer, hidden_states, block_states, num_finished)

        if len(core) > 0:
            core_input = (hidden_states, block_states, num_finished)
            for pass_idx in range(self.loop_num_passes):
                message_bank = None
                if pass_idx > 0:
                    # The previous pass's block outputs: the blocks it finished after its first, and its last sum.
                    finished = block_states[:, :, core_input[2] + 1 : num_finished]
                    message_bank = torch.cat([finished, hidden_states.unsqueeze(2)], dim=2)
                    hidden_states, block_states, num_finished = core_input
                for core_idx, layer in enumerate(self.layers[core.start : core.stop]):
                    hidden_states, block_states, num_finished = run(
                        layer,
                        hidden_states,
                        block_states,
                        num_finished,
                        message_bank,
                        # A copy: a compiled layer's input must not be a view of the larger parameter.
                        self.message_queries[core_idx].clone() if message_bank is not None else None,
                    )

        for layer in self.layers[core.stop :]:
            hidden_states, block_states, num_finished = run(layer, hidden_states, block_states, num_finished)

        block_mask = block_slots < num_finished
        return TextModelOutput(last_hidden_state=self.norm(block_states, block_mask, hidden_states))


class Inkling2LoopForCausalLM(Inkling2LoopPreTrainedModel):
    """Text-only Inkling 2 Mini. Built from the composite checkpoint config; the vision tower is not loaded."""

    def __init__(self, config: Inkling2LoopConfig | Inkling2LoopTextConfig) -> None:
        super().__init__(config)
        text_config = getattr(config, "text_config", config)
        self.model = Inkling2LoopModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = VanillaOutputLinear(
            text_config.hidden_size, text_config.num_logits, softcap=text_config.final_logit_softcapping
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def num_active_params(self) -> int:
        """Parameters a text token multiplies through, counting the loop core once per pass."""
        core_layers = self.model.layers[self.model.loop_core.start : self.model.loop_core.stop]
        core_params = [param for layer in core_layers for param in layer.parameters()]
        extra_passes = self.model.loop_num_passes - 1
        return count_active_params(self) + extra_passes * count_active_params(self, core_params)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | Tensor = 0,
        temperature: Tensor | None = None,
        support: TokenSupport | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
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
            support=support,
        )


__all__ = [
    "Inkling2DecoderLayer",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopModel",
    "Inkling2LoopPreTrainedModel",
]
