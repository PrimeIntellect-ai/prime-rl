"""DeepSeek-V4.1 decoder stack, model and causal-LM head.

The residual is `hc_mult` parallel streams `(batch, seq, hc_mult, hidden)` as in V4, but the
streams are collapsed one sublayer late: each sublayer's mHC gates come from the streams entering
it, its `post`/`comb` gates write its own output back, and its `pre` gate collapses the streams
for the next sublayer. The first attention reads stream 0, and the last FFN's `pre` gate produces
the final hidden state, so there is no separate head collapse.

Engram lookups are added into the streams before `engram_layer_ids`. They sit between decoder
layers, outside the compiled and checkpointed blocks, because their table lookup is a pair of
all-to-alls with data-dependent split sizes.
"""

from __future__ import annotations

from torch import Tensor, nn

from prime_rl.trainer.models.base import CPSupport, PrimeModel
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_state_dict_
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.deepseek_v41.attention import DeepseekV41Attention, PackedContext, SharedAttnState
from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig
from prime_rl.trainer.models.deepseek_v41.converting_deepseek_v41 import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.deepseek_v41.engram import DeepseekV41Engram, EngramHasher
from prime_rl.trainer.models.deepseek_v41.hyperconnections import (
    DeepseekV41HyperConnection,
    collapse_streams,
    identity_pre_mix,
)
from prime_rl.trainer.models.deepseek_v41.quantize import quantize_state_dict_
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.cp import CPContext


class DeepseekV41DecoderLayer(nn.Module):
    """One hyper-connected block: attention then MoE, each between delayed mHC gates.

    Takes and returns the residual streams, the `pre` gate collapsing them for this block's
    attention, and the shared attention state (as a flat tuple, so FSDP and activation
    checkpointing see every tensor).
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, rotary_emb: DeepseekV4RotaryEmbedding):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(config, layer_idx, rotary_emb)
        self.mlp = DeepseekV4MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)

    def forward(
        self,
        mhc_states: Tensor,
        pre_mix: Tensor,
        compressed_kv: Tensor | None,
        index_k: Tensor | None,
        top_k_indices: Tensor | None,
        candidates: Tensor | None,
        routed_experts: Tensor | None = None,
        *,
        packed: PackedContext,
    ) -> tuple[Tensor, ...]:
        state = SharedAttnState(compressed_kv, index_k, top_k_indices, candidates)

        attn_pre, post, comb, attn_in, streams = self.attn_hc.gates_and_collapse(mhc_states, pre_mix)
        attn_out, state = self.self_attn(self.input_layernorm(attn_in), packed=packed, state=state)
        mhc_states = self.attn_hc.update_states(post, comb, attn_out, streams)

        ffn_pre, post, comb, ffn_in, streams = self.ffn_hc.gates_and_collapse(mhc_states, attn_pre)
        mlp_out = self.mlp(self.post_attention_layernorm(ffn_in), routed_experts=routed_experts)
        mhc_states = self.ffn_hc.update_states(post, comb, mlp_out, streams)
        return (mhc_states, ffn_pre, *state.as_tuple())


# fp32 in the published checkpoint; mirrors V4's list.
KEEP_IN_FP32_MODULES = ("attn_hc", "ffn_hc", "sinks", "selection_bias")


class DeepseekV41TextModel(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(config, layer_idx, self.rotary_emb)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.engram_hasher = EngramHasher(config) if config.active_engram_layer_ids else None
        self.engrams = nn.ModuleDict(
            {
                str(layer_idx): DeepseekV41Engram(config, config.engram_layer_ids.index(layer_idx))
                for layer_idx in config.active_engram_layer_ids
            }
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.cp_context = CPContext()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        """Same packed-batch contract as V4: `seq_lens` describes the whole row (pre-shard under CP),
        `input_ids` / `position_ids` this rank's shard of it."""
        cp_rank, cp_world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
        assert seq_lens_are_pre_shard == (cp_world_size > 1), (
            f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} disagrees with cp_world_size={cp_world_size}"
        )
        inputs_embeds = self.embed_tokens(input_ids)
        packed = PackedContext.build(
            config=self.config,
            seq_lens=seq_lens,
            device=inputs_embeds.device,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        )
        assert packed.position_ids.shape[1] == inputs_embeds.shape[1], (
            f"seq_lens covers {packed.position_ids.shape[1]} tokens per rank, but this rank holds {inputs_embeds.shape[1]}"
        )
        packed.check_position_ids(position_ids)

        hash_ids = self.engram_hasher(input_ids, packed.tok_doc_start) if self.engram_hasher is not None else None
        for engram in self.engrams.values():
            engram.prefetch(hash_ids[:, engram.engram_idx])

        mhc_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = identity_pre_mix(mhc_states)
        shared: tuple[Tensor | None, ...] = SharedAttnState().as_tuple()
        for layer_idx, decoder_layer in enumerate(self.layers):
            engram = self.engrams[str(layer_idx)] if str(layer_idx) in self.engrams else None
            if engram is not None:
                mhc_states = engram(mhc_states, hash_ids[:, engram.engram_idx])
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            mhc_states, pre_mix, *shared = decoder_layer(
                mhc_states, pre_mix, *shared, routed_experts_layer, packed=packed
            )
        return self.norm(collapse_streams(mhc_states, pre_mix))


class DeepseekV41ForCausalLM(PrimeModel):
    def __init__(self, config: DeepseekV41Config | DeepseekV41TextConfig):
        super().__init__(config)
        # The published config nests the text model beside a vision tower that is not trained, so
        # the composite config only contributes its `text_config`.
        text_config = config.text_config if isinstance(config, DeepseekV41Config) else config
        self.model = DeepseekV41TextModel(text_config)
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)

    @classmethod
    def cp_support(cls, config: DeepseekV41Config | DeepseekV41TextConfig) -> CPSupport:
        return CPSupport(
            frozenset({"ring"}),
            "queries are sharded and the single-head KV latents, compressed entries and index keys are "
            "all-gathered, which needs contiguous shards",
        )

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(module_name in name for module_name in KEEP_IN_FP32_MODULES)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config: DeepseekV41Config | DeepseekV41TextConfig):
        return conversion_chain(config)

    def skip_weight_transfer(self, name: str) -> bool:
        if ".engrams." not in name or not name.endswith(".embed.weight"):
            return False
        if self.get_parameter(name).requires_grad:
            raise ValueError(
                f"{name} is too large to broadcast to the inference engine; set model.freeze_engram_tables = true"
            )
        return True

    def to_inference_format(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """vLLM serves V4.1 from its fp8 / fp4 checkpoint, so weights go back into that format."""
        quantize_state_dict_(state_dict)
        return state_dict

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Dequantize the published fp8 / fp4 weights to bf16, then rename."""
        dequantize_state_dict_(state_dict)
        return super().convert_to_prime(state_dict)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> PrimeLmOutput:
        hidden_states = self.model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard, routed_experts)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    def prune_to_pipeline_stage(self, layer_ids: range, *, first: bool, last: bool) -> None:
        """Keep only this pipeline stage's decoder layers (and engrams), the embedding on the first
        stage and the final norm and head on the last."""
        from prime_rl.trainer.pipeline import StageLayers

        model = self.model
        model.layers = StageLayers({idx: model.layers[idx] for idx in layer_ids})
        model.engrams = nn.ModuleDict({key: engram for key, engram in model.engrams.items() if int(key) in layer_ids})
        if len(model.engrams) == 0:
            model.engram_hasher = None
        if not first:
            model.embed_tokens = None
        if not last:
            model.norm = None
            self.lm_head = None

    def pipeline_stage_forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """One pipeline stage. The first stage takes `(input_ids, position_ids, labels)`; later
        stages take the residual streams, the `pre` gate and the shared attention state (an empty
        tensor where it is still unset) followed by those three. The last stage returns the summed
        cross-entropy, the others what the next stage takes.

        Each call reads its micro-batch's document lengths from `pipeline_seq_lens`, host tensors
        queued in micro-batch order (a stage runs its forwards in that order), so building the
        packing context never waits on the GPU."""
        model = self.model
        seq_lens = self.pipeline_seq_lens.pop(0)
        if model.embed_tokens is not None:
            input_ids, position_ids, labels = inputs
            mhc_states = (
                model.embed_tokens(input_ids).unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
            )
            pre_mix = identity_pre_mix(mhc_states)
            shared: list[Tensor | None] = list(SharedAttnState().as_tuple())
        else:
            mhc_states, pre_mix, *shared, input_ids, position_ids, labels = inputs
            shared = [None if tensor.numel() == 0 else tensor for tensor in shared]
        # Positions were checked against `seq_lens` on the host when the micro-batches were stacked.
        packed = PackedContext.build(
            config=model.config, seq_lens=seq_lens, device=input_ids.device, cp_rank=0, cp_world_size=1
        )
        hash_ids = model.engram_hasher(input_ids, packed.tok_doc_start) if model.engram_hasher is not None else None
        for engram in model.engrams.values():
            engram.prefetch(hash_ids[:, engram.engram_idx])
        for name, decoder_layer in model.layers.named_children():
            engram = model.engrams[name] if name in model.engrams else None
            if engram is not None:
                mhc_states = engram(mhc_states, hash_ids[:, engram.engram_idx])
            mhc_states, pre_mix, *shared = decoder_layer(mhc_states, pre_mix, *shared, None, packed=packed)
        if self.lm_head is not None:
            hidden_states = model.norm(collapse_streams(mhc_states, pre_mix))
            # One element per micro-batch: the schedule concatenates the last stage's outputs.
            return self.lm_head(hidden_states, labels)["loss"].reshape(1)
        shared = [mhc_states.new_empty(0) if tensor is None else tensor for tensor in shared]
        return (mhc_states, pre_mix, *shared, input_ids, position_ids, labels)

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, (DeepseekV4RotaryEmbedding, EngramHasher)):
                module.init_buffers_post_meta()
            elif isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()


__all__ = ["DeepseekV41DecoderLayer", "DeepseekV41ForCausalLM", "DeepseekV41TextModel"]
