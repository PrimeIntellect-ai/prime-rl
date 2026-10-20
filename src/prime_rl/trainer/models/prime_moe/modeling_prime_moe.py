"""The pretraining MoE (`prime_moe`), pretraining's own model, so architecture work leaves the shared MoE models
unchanged. Module names follow `glm4_moe`, so checkpoints keep that state-dict layout and its conversion."""

from typing import Optional

import torch
from torch import Tensor, nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import conversion_chain
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.hyper_connection import HyperConnection
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.trainer.models.prime_moe.attention import PrimeFlashAttention
from prime_rl.trainer.models.prime_moe.byte_window import ByteWindowEmbedding
from prime_rl.trainer.models.prime_moe.configuration_prime_moe import PrimeMoeConfig
from prime_rl.trainer.models.prime_moe.moe import PrimeMoE, PrimeRouter, SituGlu
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding import NGramEmbedding
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class PrimeMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PrimeMoeConfig, layer_idx: int):
        super().__init__()
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )
        if config.attn_output_gate or config.exclusive_attention or config.hc_qkv_reads or config.hc_qkv_conv:
            self.self_attn = PrimeFlashAttention(
                attn_config,
                flash_attn_version=int(config._attn_implementation.removeprefix("flash_attention_")),
                exclusive=config.exclusive_attention,
                output_gate=config.attn_output_gate,
            )
        else:
            self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](attn_config)

        if layer_idx >= config.first_k_dense_replace:
            expert_activation = SituGlu() if config.expert_activation == "situ_glu" else config.hidden_act
            shared_expert = None
            if config.shared_expert_intermediate_size > 0:
                shared_expert = FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.shared_expert_intermediate_size,
                    expert_type="gated",
                    activation=expert_activation,
                )
            self.mlp = PrimeMoE(
                router=PrimeRouter(
                    dim=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_null_experts=config.n_null_experts,
                    top_k=config.num_experts_per_tok,
                    route_norm=config.norm_topk_prob,
                    route_scale=config.routed_scaling_factor,
                    bias_on_logits=config.router_bias_on_logits,
                ),
                experts=GroupedExperts(
                    dim=config.latent_dim or config.hidden_size,
                    hidden_dim=config.moe_intermediate_size,
                    num_experts=config.n_routed_experts,
                    expert_type="gated",
                    activation=expert_activation,
                ),
                shared_expert=shared_expert,
                dim=config.hidden_size,
                latent_dim=config.latent_dim,
                latent_norm=config.latent_norm,
                norm_eps=config.rms_norm_eps,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size, hidden_dim=config.intermediate_size, activation=config.hidden_act
            )

        self.gated_residual = config.hc_count > 1
        if self.gated_residual:
            connection_args = {
                "hidden_size": config.hidden_size,
                "stream_count": config.hc_count,
                "low_rank": config.hc_lowrank,
                "norm_eps": config.rms_norm_eps,
                "stream_width": config.hc_stream_width,
            }
            self.attention_read_count = 3 if config.hc_qkv_reads else 1
            self.parallel_block = config.hc_parallel_block
            if self.parallel_block:
                # Attention and MLP read the same streams (one read, or their own reads with `separate_reads`) and
                # both write to all streams, each with its own scalars.
                separate = self.parallel_block == "separate_reads"
                self.hyper_connection = HyperConnection(
                    **connection_args,
                    read_count=self.attention_read_count + separate,
                    write_count=2,
                    conv_kernel_size=config.hc_qkv_conv,
                )
            else:
                self.attn_hyper_connection = HyperConnection(
                    **connection_args,
                    read_count=self.attention_read_count,
                    conv_kernel_size=config.hc_qkv_conv,
                    write_mode=config.hc_write,
                )
                moe_reads = config.hc_moe_reads and isinstance(self.mlp, PrimeMoE)
                self.mlp_hyper_connection = HyperConnection(
                    **connection_args, read_count=2 if moe_reads else 1, write_mode=config.hc_write
                )
        else:
            self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
            self.post_attention_layernorm = RMSNorm(
                RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if self.gated_residual and self.parallel_block:
            reads, residual_state = self.hyper_connection.read(hidden_states)
            attention_output, _ = self.self_attn(
                hidden_states=self.hyper_connection.attention_inputs(reads[: self.attention_read_count], cu_seqlens),
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            mlp_output = self.mlp(reads[-1] if self.parallel_block == "separate_reads" else reads[0])
            return self.hyper_connection.combine((attention_output, mlp_output), residual_state)

        if self.gated_residual:
            block_input, residual_state = self.attn_hyper_connection.mix(hidden_states, cu_seqlens)
            block_output, _ = self.self_attn(
                hidden_states=block_input,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hidden_states = self.attn_hyper_connection.combine(block_output, residual_state)
            block_input, residual_state = self.mlp_hyper_connection.mix(hidden_states)
            block_output = self.mlp(block_input)
            return self.mlp_hyper_connection.combine(block_output, residual_state)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class PrimeMoePreTrainedModel(PreTrainedModelPrimeRL):
    config: PrimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PrimeMoeDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith("mlp.router.selection_bias")

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        # The expert state-dict layout of glm4_moe, so its conversion applies.
        return conversion_chain(config)


class PrimeMoeModel(PrimeMoePreTrainedModel):
    def __init__(self, config: PrimeMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Per-stream inputs: the first stream is the token embedding, the others extra token tables, and with
        # `ngram`, `bytes` or several of them joined by `_` (e.g. `ngram_bytes`, `ngram_ngram`) the last stream(s) hashed
        # token n-grams and / or the token's preceding bytes. A repeated kind is an independent copy (own parameters;
        # every n-gram stream hashes with its own layout). With `hc_byte_add`, an own byte embedding is added to the
        # input of every token-table stream (`tables`), every n-gram stream (`ngrams`) or both (`all`).
        kinds = [] if config.hc_stream_inputs in ("copy", "tables") else config.hc_stream_inputs.split("_")
        tables = 0 if config.hc_stream_inputs == "copy" else config.hc_count - 1 - len(kinds)
        self.extra_tables = tables
        self.extra_embed_tokens = nn.Embedding(config.vocab_size, tables * config.hidden_size) if tables else None

        def ngram_stream(layout: int) -> NGramEmbedding:
            return NGramEmbedding(
                embedding_dim=config.hidden_size,
                ngram_size=3,
                heads_per_ngram=2,
                ngram_vocab_size=config.hc_ngram_vocab_size,
                token_vocab_size=config.vocab_size,
                eos_token_id=config.eos_token_id,
                vocab_size_divisor=128,
                ngram_layer_index=layout,
            )

        def byte_stream() -> ByteWindowEmbedding:
            return ByteWindowEmbedding(
                dim=config.hidden_size, window=config.hc_byte_window, table_path=config.hc_byte_table
            )

        self.ngram_embed_tokens = ngram_stream(0) if "ngram" in kinds else None
        self.ngram_embed_tokens_2 = ngram_stream(1) if kinds.count("ngram") >= 2 else None
        self.ngram_embed_tokens_3 = ngram_stream(2) if kinds.count("ngram") >= 3 else None
        self.byte_embed_tokens = byte_stream() if "bytes" in kinds else None
        self.byte_embed_tokens_2 = byte_stream() if kinds.count("bytes") >= 2 else None
        ngrams = kinds.count("ngram")
        added_to = {None: 0, "tables": 1 + tables, "ngrams": ngrams, "all": 1 + tables + ngrams}[config.hc_byte_add]
        self.byte_add_embed_tokens = nn.ModuleList(byte_stream() for _ in range(added_to)) if added_to else None
        self.layers = nn.ModuleList(
            [PrimeMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.hc_count > 1:
            self.hyper_connection_mixer = HyperConnection(
                hidden_size=config.hidden_size,
                stream_count=config.hc_count,
                low_rank=config.hc_lowrank,
                norm_eps=config.rms_norm_eps,
                with_residual_injection=False,
                stream_width=config.hc_stream_width,
            )
        else:
            self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        if isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        self.rotary_emb = RotaryEmbedding(
            RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings, rope_type=rope_type, model_config=config
            )
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        inputs_embeds = self.embed_tokens(input_ids)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        if self.config.hc_stream_inputs == "copy":
            # Gated Residual: every residual stream starts from the token embedding (its chunks, for narrower streams).
            stream_width = self.config.hc_stream_width or self.config.hidden_size
            hidden_states = inputs_embeds.repeat(1, 1, self.config.hc_count * stream_width // self.config.hidden_size)
        else:
            table_streams = [inputs_embeds]
            if self.extra_embed_tokens is not None:
                table_streams += self.extra_embed_tokens(input_ids).chunk(self.extra_tables, dim=-1)
            ngram_streams = [
                stream(input_ids, cu_seqlens)
                for stream in (self.ngram_embed_tokens, self.ngram_embed_tokens_2, self.ngram_embed_tokens_3)
                if stream is not None
            ]
            byte_streams = [
                stream(input_ids, cu_seqlens)
                for stream in (self.byte_embed_tokens, self.byte_embed_tokens_2)
                if stream is not None
            ]
            if self.config.hc_byte_add is not None:
                added = [stream(input_ids, cu_seqlens) for stream in self.byte_add_embed_tokens]
                if self.config.hc_byte_add in ("tables", "all"):
                    table_streams = [stream + bytes_ for stream, bytes_ in zip(table_streams, added)]
                    added = added[len(table_streams) :]
                if self.config.hc_byte_add in ("ngrams", "all"):
                    ngram_streams = [stream + bytes_ for stream, bytes_ in zip(ngram_streams, added)]
            streams = table_streams + ngram_streams + byte_streams
            hidden_states = torch.cat(streams, dim=-1)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        if self.config.hc_count > 1:
            hidden_states, _ = self.hyper_connection_mixer(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class PrimeMoeForCausalLM(PrimeMoePreTrainedModel):
    def __init__(self, config: PrimeMoeConfig):
        super().__init__(config)
        self.model = PrimeMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: Optional[torch.Tensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        """``labels`` and ``temperature`` go to prime-rl's wrapped LM head, which then returns per-token logprobs
        and entropy; without them it returns logits."""
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return self.lm_head(outputs.last_hidden_state, labels, temperature=temperature)

    def init_buffers_post_meta(self):
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)
        model = self.model
        for stream in (model.ngram_embed_tokens, model.ngram_embed_tokens_2, model.ngram_embed_tokens_3):
            if stream is not None:
                stream.reset_parameters()
        for stream in (model.byte_embed_tokens, model.byte_embed_tokens_2, *(model.byte_add_embed_tokens or ())):
            if stream is not None:
                stream.reset_buffers()
