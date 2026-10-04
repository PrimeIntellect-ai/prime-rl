import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.trainer.models.llama.configuration_llama import LlamaConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=False,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.self_attn = ATTN_IMPL2CLASS[config.attn_implementation](attn_config)
        self.mlp = FeedForward(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
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


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = RotaryEmbedding(config.rope_parameters, config.head_dim, config.max_position_embeddings)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=hidden_states.device),
            total_tokens=None if seq_lens_are_pre_shard else hidden_states.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        return self.norm(hidden_states)


class LlamaForCausalLM(PrimeModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

    # Dense models use the HF key names for training too, so there is no separate PrimeRL format.
    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return False

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
    ) -> PrimeLmOutput:
        hidden_states = self.model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)
