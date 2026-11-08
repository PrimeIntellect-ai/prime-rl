from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import (
    flash_attn_2_varlen_op,
    flash_attn_3_varlen_op,
    quadratic_attention_flops,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.nanochat.configuration_nanochat import NanochatConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

VE_GATE_CHANNELS = 32


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_value_embedding(layer_idx: int, num_layers: int) -> bool:
    return layer_idx % 2 == (num_layers - 1) % 2


def window_sizes(config: NanochatConfig) -> list[int]:
    """Attended tokens to the left per layer: the pattern tiled across layers, S = half context, L = full context,
    the last layer always L."""
    pattern = config.window_pattern.upper()
    long_window, short_window = config.max_position_embeddings, config.max_position_embeddings // 2
    windows = [
        {"L": long_window, "S": short_window}[pattern[i % len(pattern)]] for i in range(config.num_hidden_layers)
    ]
    windows[-1] = long_window
    return windows


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class NanochatAttention(nn.Module):
    def __init__(self, config: NanochatConfig, layer_idx: int, window: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = window
        self.attn_func = {"flash_attention_2": flash_attn_2_varlen_op, "flash_attention_3": flash_attn_3_varlen_op}.get(
            getattr(config, "_attn_implementation", "flash_attention_3"), flash_attn_3_varlen_op
        )
        self.c_q = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ve_gate = (
            nn.Linear(VE_GATE_CHANNELS, self.num_kv_heads, bias=False)
            if has_value_embedding(layer_idx, config.num_hidden_layers)
            else None
        )

    def attention_flops(self, seq_len: int) -> int:
        return quadratic_attention_flops(
            num_heads=self.num_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            seq_len=seq_len,
            attended_tokens=min(self.window, seq_len),
        )

    def forward(
        self,
        x: Tensor,
        value_embedding: Tensor | None,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        batch, tokens, _ = x.shape
        q = self.c_q(x).view(batch, tokens, self.num_heads, self.head_dim)
        k = self.c_k(x).view(batch, tokens, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).view(batch, tokens, self.num_kv_heads, self.head_dim)
        if value_embedding is not None:
            value_embedding = value_embedding.view(batch, tokens, self.num_kv_heads, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * value_embedding
        q, k = rms_norm(apply_rotary(q, cos, sin)), rms_norm(apply_rotary(k, cos, sin))
        q, k, v = q.reshape(-1, *q.shape[2:]), k.reshape(-1, *k.shape[2:]), v.reshape(-1, *v.shape[2:])
        if q.is_cuda:
            out = self.attn_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, None, True, (self.window, 0))
        else:
            out = self._sdpa(q, k, v, cu_seqlens)
        return self.c_proj(out.contiguous().view(batch, tokens, -1))

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Packed causal sliding-window attention for CPU (flash attention needs a GPU)."""
        outputs = []
        for begin, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            positions = torch.arange(end - begin, device=q.device)
            mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)) & (
                positions.unsqueeze(1) - positions.unsqueeze(0) <= self.window
            )
            outputs.append(
                F.scaled_dot_product_attention(
                    q[begin:end].transpose(0, 1),
                    k[begin:end].transpose(0, 1),
                    v[begin:end].transpose(0, 1),
                    attn_mask=mask,
                ).transpose(0, 1)
            )
        return torch.cat(outputs, dim=0)


class NanochatMLP(nn.Module):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class NanochatBlock(GradientCheckpointingLayer):
    def __init__(self, config: NanochatConfig, layer_idx: int, window: int) -> None:
        super().__init__()
        self.attn = NanochatAttention(config, layer_idx, window)
        self.mlp = NanochatMLP(config)
        self.value_embed = (
            nn.Embedding(config.vocab_size, config.num_key_value_heads * config.head_dim)
            if has_value_embedding(layer_idx, config.num_hidden_layers)
            else None
        )
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        state: Tensor,
        input_ids: Tensor,
        x0: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        state = self.resid_lambda.to(state.dtype) * state + self.x0_lambda.to(state.dtype) * x0
        value_embedding = self.value_embed(input_ids) if self.value_embed is not None else None
        state = state + self.attn(rms_norm(state), value_embedding, cos, sin, cu_seqlens, max_seqlen)
        return state + self.mlp(rms_norm(state))


class NanochatPreTrainedModel(PreTrainedModelPrimeRL):
    config: NanochatConfig
    config_class = NanochatConfig
    base_model_prefix = "model"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Dense model with identical key names in HF and PrimeRL format.
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


class NanochatModel(NanochatPreTrainedModel):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__(config)
        windows = window_sizes(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NanochatBlock(config, layer_idx, windows[layer_idx]) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, elementwise_affine=False)
        cos, sin = self._rotary(config, device=None)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.gradient_checkpointing = False
        self.post_init()

    @staticmethod
    def _rotary(config: NanochatConfig, device) -> tuple[Tensor, Tensor]:
        channels = torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channels / config.head_dim))
        time = torch.arange(config.max_position_embeddings * 10, dtype=torch.float32, device=device)
        frequencies = torch.outer(time, inv_freq)
        return frequencies.cos().bfloat16()[None, :, None, :], frequencies.sin().bfloat16()[None, :, None, :]

    def init_rotary_buffers(self) -> None:
        cos, sin = self._rotary(self.config, device=self.cos.device)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        positions = position_ids[0]
        cos, sin = self.cos[:, positions, :, :], self.sin[:, positions, :, :]
        x0 = rms_norm(inputs_embeds)
        state = x0
        for block in self.layers[: self.config.num_hidden_layers]:
            state = block(state, input_ids, x0, cos, sin, cu_seqlens, max_seqlen)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(state))


class NanochatForCausalLM(NanochatPreTrainedModel, GenerationMixin):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__(config)
        self.model = NanochatModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.softcap = config.logit_softcap
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: Optional[torch.Tensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
        self.model.init_rotary_buffers()

    def _init_weights(self, module: nn.Module) -> None:
        pass

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """nanochat's ``init_weights``, deterministic in ``config.init_seed`` so every replica builds the same model."""
        if next(self.parameters()).is_meta:
            return
        hidden_size = self.config.hidden_size
        bound = (3 / hidden_size) ** 0.5
        generator = torch.Generator(device=_local(next(self.parameters())).device).manual_seed(self.config.init_seed)
        _local(self.model.embed_tokens.weight).normal_(0.0, 1.0, generator=generator)
        _local(self.lm_head.weight).normal_(0.0, 0.001, generator=generator)
        for block in self.model.layers:
            for parameter in (
                block.attn.c_q.weight,
                block.attn.c_k.weight,
                block.attn.c_v.weight,
                block.mlp.c_fc.weight,
            ):
                _local(parameter).uniform_(-bound, bound, generator=generator)
            _local(block.attn.c_proj.weight).zero_()
            _local(block.mlp.c_proj.weight).zero_()
            _local(block.resid_lambda).fill_(1.0)
            _local(block.x0_lambda).fill_(0.1)
            if block.value_embed is not None:
                _local(block.value_embed.weight).uniform_(-bound, bound, generator=generator)
                _local(block.attn.ve_gate.weight).zero_()

    def estimate_flops(self) -> int:
        """FLOPs per token, forward + backward, in nanochat's ``estimate_flops`` convention: 6 per matmul parameter
        plus the attention term."""
        excluded = {id(self.model.embed_tokens.weight)}
        for block in self.model.layers:
            excluded.update({id(block.resid_lambda), id(block.x0_lambda)})
            if block.value_embed is not None:
                excluded.add(id(block.value_embed.weight))
        matmul_parameters = sum(p.numel() for p in self.parameters() if id(p) not in excluded)
        tokens = self.config.max_position_embeddings
        attention = sum(block.attn.attention_flops(tokens) for block in self.model.layers) // tokens
        return 6 * matmul_parameters + attention


__all__ = ["NanochatForCausalLM", "NanochatModel", "NanochatPreTrainedModel"]
