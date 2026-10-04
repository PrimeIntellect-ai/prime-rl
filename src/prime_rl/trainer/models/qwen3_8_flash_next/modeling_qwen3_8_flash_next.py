import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import CPSupport, PrimeModel
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
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
from prime_rl.trainer.models.qwen3_8_flash_next.hyper_connection import HyperConnection
from prime_rl.trainer.models.qwen3_8_flash_next.moe import SigmoidOutputGatedMoE
from prime_rl.trainer.models.qwen3_8_flash_next.position_learning import PositionLearningEnhancement
from prime_rl.trainer.models.qwen3_8_flash_next.rotary_embedding import RotaryEmbedding
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


class Qwen3_8FlashNextTextModel(nn.Module):
    def __init__(self, config: Qwen3_8FlashNextTextConfig) -> None:
        super().__init__()
        self.hc_count = config.hc_count
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
        self.rotary_emb = RotaryEmbedding(config.rope_parameters, config.head_dim)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        """``routed_experts`` (``[batch, seq, num_layers, top_k]``) replays the inference router's choices."""
        inputs_embeds = self.embed_tokens(input_ids)
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(
            seq_lens.to(inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds.repeat(1, 1, self.hc_count)
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
        return hidden_states


class Qwen3_8FlashNextModel(nn.Module):
    """Composite (``qwen4_exp``) wrapper; only the language model is trained."""

    def __init__(self, config: Qwen3_8FlashNextConfig) -> None:
        super().__init__()
        self.language_model = Qwen3_8FlashNextTextModel(config.text_config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        return self.language_model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard, routed_experts)


class Qwen3_8FlashNextForCausalLM(PrimeModel):
    def __init__(self, config: Qwen3_8FlashNextConfig | Qwen3_8FlashNextTextConfig) -> None:
        super().__init__(config)
        if isinstance(config, Qwen3_8FlashNextConfig):
            text_config = config.text_config
            self.model = Qwen3_8FlashNextModel(config)
        else:
            text_config = config
            self.model = Qwen3_8FlashNextTextModel(config)
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)

    @classmethod
    def cp_support(cls, config: Qwen3_8FlashNextConfig | Qwen3_8FlashNextTextConfig) -> CPSupport:
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
    def conversion_chain(cls, config: Qwen3_8FlashNextConfig | Qwen3_8FlashNextTextConfig):
        return conversion_chain(config)

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


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextModel",
    "Qwen3_8FlashNextTextModel",
]
