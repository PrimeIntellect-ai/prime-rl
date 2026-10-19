"""DeepSeek-V4.1 decoder stack, model and causal-LM head.

The residual is `hc_mult` parallel streams `(batch, seq, hc_mult, hidden)` as in V4, but the
streams are collapsed one sublayer late: each sublayer's mHC gates come from the streams entering
it, its `post`/`comb` gates write its own output back, and its `pre` gate collapses the streams
for the next sublayer. The first attention reads stream 0, and the last FFN's `pre` gate produces
the final hidden state, so there is no separate head collapse.

Engram lookups are added into the streams before `engram_layer_ids`. They sit between decoder
layers, outside the checkpointed blocks, because their table lookup is a pair of all-to-alls with
data-dependent split sizes.
"""

from __future__ import annotations

from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_weight
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
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


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
    ) -> tuple[Tensor | None, ...]:
        state = SharedAttnState(compressed_kv, index_k, top_k_indices, candidates)

        attn_pre, post, comb = self.attn_hc(mhc_states)
        attn_out, state = self.self_attn(
            self.input_layernorm(collapse_streams(mhc_states, pre_mix)), packed=packed, state=state
        )
        mhc_states = self.attn_hc.update_states(post, comb, attn_out, mhc_states)

        ffn_pre, post, comb = self.ffn_hc(mhc_states)
        mlp_out = self.mlp(
            self.post_attention_layernorm(collapse_streams(mhc_states, attn_pre)), routed_experts=routed_experts
        )
        mhc_states = self.ffn_hc.update_states(post, comb, mlp_out, mhc_states)
        return (mhc_states, ffn_pre, *state.as_tuple())


# fp32 in the published checkpoint. Mirrors V4's list, with `e_score_correction_bias` under the
# `selection_bias` name prime-rl's router keeps it as; V4.1 has no `hc_head`.
KEEP_IN_FP32_MODULES = ("attn_hc", "ffn_hc", "sinks", "selection_bias")


class DeepseekV41PreTrainedModel(PreTrainedModelPrimeRL):
    config: DeepseekV41Config
    config_class = DeepseekV41Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # V4.1 attention runs its own fused kernel; no transformers backend can serve it.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [r"^(mtp|vision|aligner)\.", r"^image_(start|end|newline)$"]

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        return "context parallelism is not implemented for DeepSeek-V4.1"

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        init_std = self.config.get_text_config().initializer_range
        if isinstance(module, (DeepseekV41Attention, DeepseekV41HyperConnection, DeepseekV41Engram)):
            module.init_weights(init_std)
        elif isinstance(module, DeepseekV4MoE):
            module.init_weights(init_std, module.tokens_per_expert.device)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            module.init_buffers_post_meta()

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
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Dequantize the fp8 engram tables to bf16, then rename.

        Expects the bf16 checkpoint (`PrimeIntellect/DeepSeek-V4.1-Flash-bf16`), where everything
        except the engram tables is already bf16. The tables stay fp8 there because vLLM serves them
        only in fp8.
        """
        for key in [k for k in state_dict if k.endswith(".scale")]:
            weight_key = key.removesuffix(".scale") + ".weight"
            if ".engram.embed." not in key:
                raise ValueError(
                    f"{weight_key} is quantized; load the bf16 checkpoint PrimeIntellect/DeepSeek-V4.1-Flash-bf16"
                )
            state_dict[weight_key] = dequantize_weight(state_dict[weight_key], state_dict.pop(key))
        return super().convert_to_prime(state_dict)


class DeepseekV41Model(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig, tokenizer_path: str | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Shared by every attention layer instead of passing RoPE tensors through forward, which FSDP casts to bf16.
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(config, layer_idx, self.rotary_emb)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.engram_hasher = EngramHasher(config, tokenizer_path) if config.active_engram_layer_ids else None
        self.engrams = nn.ModuleDict(
            {
                str(layer_idx): DeepseekV41Engram(config, config.engram_layer_ids.index(layer_idx))
                for layer_idx in config.active_engram_layer_ids
            }
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> MoeModelOutputWithPast:
        """
        input_ids (`torch.LongTensor` of shape `(1, sequence_length)`):
            One packed row of token ids. Besides the embedding, the engram hash reads them.
        position_ids (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*):
            Each token's position within its own document. Only checked: every position is derived
            from `seq_lens`, so the two cannot disagree.
        routed_experts (`torch.LongTensor` of shape `(1, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row. Clips the sliding window, the compressed entries,
            the indexer's picks and the engram look-back at document boundaries.
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Only `False` is accepted: V4.1 does not support context parallelism.
        """
        assert not seq_lens_are_pre_shard, "DeepSeek-V4.1 does not support context parallelism"
        inputs_embeds = self.embed_tokens(input_ids)
        packed = PackedContext.build(config=self.config, seq_lens=seq_lens, device=inputs_embeds.device)
        assert packed.position_ids.shape[1] == inputs_embeds.shape[1], (
            f"seq_lens covers {packed.position_ids.shape[1]} tokens, but the row holds {inputs_embeds.shape[1]}"
        )
        if position_ids is not None:
            packed.check_position_ids(position_ids)

        hash_ids = self.engram_hasher(input_ids, packed.tok_doc_start) if self.engram_hasher is not None else None

        mhc_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = identity_pre_mix(mhc_states)
        shared: tuple[Tensor | None, ...] = SharedAttnState().as_tuple()
        for layer_idx, decoder_layer in enumerate(self.layers):
            if str(layer_idx) in self.engrams:
                engram = self.engrams[str(layer_idx)]
                mhc_states = engram(mhc_states, hash_ids[:, engram.engram_idx])
            routed_experts_layer = (
                routed_experts[:, :, layer_idx, :].contiguous() if routed_experts is not None else None
            )
            mhc_states, pre_mix, *shared = decoder_layer(
                mhc_states, pre_mix, *shared, routed_experts_layer, packed=packed
            )
        return MoeModelOutputWithPast(last_hidden_state=self.norm(collapse_streams(mhc_states, pre_mix)))


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV41Config | DeepseekV41TextConfig):
        super().__init__(config)
        # The published config nests the text model beside a vision tower that is not trained, so
        # the composite config only contributes its `text_config`. The engram hash is built from the
        # checkpoint's tokenizer, which sits next to the config.
        text_config = config.get_text_config()
        self.model = DeepseekV41Model(text_config, tokenizer_path=config.name_or_path)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        logits_to_keep: int | Tensor = 0,
        temperature: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        """
        labels (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(1, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are given.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row. Required, so a caller that forgets it fails
            instead of silently getting a window that spans neighbouring documents.
        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
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

    def init_buffers_post_meta(self) -> None:
        # Rebuilt here are the buffers no checkpoint carries: the shared rotary tables, the engram
        # hash tables and `tokens_per_expert`. The router's `selection_bias` comes from the checkpoint.
        for module in self.modules():
            if isinstance(module, (DeepseekV4RotaryEmbedding, EngramHasher)):
                module.init_buffers_post_meta()
            elif isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()


__all__ = [
    "DeepseekV41DecoderLayer",
    "DeepseekV41ForCausalLM",
    "DeepseekV41Model",
    "DeepseekV41PreTrainedModel",
]
