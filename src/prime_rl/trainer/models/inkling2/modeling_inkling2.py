"""Inkling 2 language model.

A decoder-only MoE transformer without a residual stream: before each sublayer a learned query mixes the
normalized token embeddings, the completed blocks of `n_layers_per_attn_res_block` layers and the running
partial block (block attention residuals). Attention is GQA with a learned relative-position bias and no
RoPE, over a sliding window in local layers and over the whole document in every sixth layer, followed by a
per-channel gate and a per-head sink. Keys, values and both sublayer outputs pass through short causal
convolutions.

Rounding points follow the serving kernels: norms, mixes, convolutions, clips, the gate and the expert
activation compute in fp32 and round once to the parameter dtype; linears round to the parameter dtype; the
router and the unembedding produce fp32.
"""

from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask, flex_attention
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from prime_rl.trainer.models.inkling2.converting_inkling2 import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.layers.activations import SoftcappedSwiglu
from prime_rl.trainer.models.layers.attn import quadratic_attention_flops
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, TokenSupport, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

ATTN_RES_EPS = 1e-6
L2_CLIP_EPS = 1e-6
# FlexAttention's kernel block. A row is padded to a multiple of it (other lengths can get a block size that
# no kernel configuration divides), and past FLEX_BUCKET to a multiple of FLEX_BUCKET, which bounds the
# number of shapes the kernel compiles for.
FLEX_BLOCK_SIZE = 128
FLEX_BUCKET = 1024

AttentionBackend = Literal["auto", "dense", "flex"]


@cache
def compiled_flex_attention() -> Callable:
    """Uncompiled, FlexAttention materializes the dense scores."""
    return torch.compile(flex_attention, dynamic=False)


def document_layout(cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
    """Each token's position within its document and its document index, for one packed row."""
    cu_seqlens = cu_seqlens.long()
    tokens = torch.arange(int(cu_seqlens[-1]), device=cu_seqlens.device)
    doc_ids = torch.searchsorted(cu_seqlens[1:], tokens, right=True)
    return tokens - cu_seqlens[doc_ids], doc_ids


def rms_norm(x: Tensor, weight: Tensor | None, eps: float, dtype: torch.dtype) -> Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(dtype)


def attn_res_mix(
    blocks: Sequence[Tensor], num_blocks: Tensor, tail: Sequence[Tensor], queries: Tensor, normalize_values: bool
) -> list[Tensor]:
    """Softmax over depth: source `n` scores `query . source_n / (rms(source_n) * sqrt(hidden))`.

    The sources, all `[tokens, hidden]`, are the first `num_blocks` (a scalar tensor) of `blocks` and then
    `tail`. Returns one fp32 `[tokens, hidden]` mixture per row of `queries`.
    """
    queries = queries.float().reshape(-1, queries.shape[-1])
    scaled_queries = queries * queries.shape[-1] ** -0.5
    values, logits = [], []
    for index, source in enumerate((*blocks, *tail)):
        value = source.float()
        inv_rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + ATTN_RES_EPS)
        logit = (value @ scaled_queries.t()) * inv_rms
        if index < len(blocks):
            logit = logit.masked_fill(num_blocks <= index, float("-inf"))
        logits.append(logit)
        values.append(value * inv_rms if normalize_values else value)
    weights = torch.softmax(torch.stack(logits, dim=-1), dim=-1)

    mixes = []
    for query_idx in range(queries.shape[0]):
        mix = weights[:, query_idx, 0:1] * values[0]
        for source_idx in range(1, len(values)):
            mix = mix + weights[:, query_idx, source_idx : source_idx + 1] * values[source_idx]
        mixes.append(mix)
    return mixes


def l2_clip(x: Tensor, threshold: float, scale: Tensor | None = None, tau: Tensor | None = None) -> Tensor:
    """Scale `x [tokens, heads, dim]` per head, clip each vector's L2 norm to `threshold` and round; `tau`
    (per token) multiplies the rounded result."""
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    out = (value * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)
    if tau is not None:
        out = (out.float() * tau.view(-1, 1, 1)).to(x.dtype)
    return out


def lagged(x: Tensor, lag: int, positions: Tensor) -> Tensor:
    """`x[t - lag]` for `x [tokens, channels]`, zero where that token precedes the start of `t`'s document."""
    if lag == 0:
        return x
    return torch.roll(x, lag, dims=0) * (positions >= lag).unsqueeze(-1)


def flex_length(seq_len: int) -> int:
    step = FLEX_BLOCK_SIZE if seq_len <= FLEX_BUCKET else FLEX_BUCKET
    return -(-seq_len // step) * step


def relative_block_mask(doc_ids: Tensor, window: int | None) -> BlockMask:
    """Causal, same-document FlexAttention mask over the last `window` tokens (all of them without one).
    Padding tokens carry document id -1."""

    def mask_mod(batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        keep = (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
        if window is not None:
            keep = keep & (q_idx - kv_idx < window)
        return keep

    length = doc_ids.numel()
    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=length, KV_LEN=length, device=doc_ids.device, BLOCK_SIZE=FLEX_BLOCK_SIZE
    )


def dense_relative_attention(
    q: Tensor, k: Tensor, v: Tensor, rel: Tensor, doc_ids: Tensor, window: int | None, query_chunk: int = 256
) -> tuple[Tensor, Tensor]:
    """Exact fp32 attention that materializes `[heads, query_chunk, tokens]` scores: the reference for the
    FlexAttention path, and the CPU path.

    `q [tokens, heads, dim]`, `k` / `v [tokens, kv_heads, dim]` and the relative logits `rel [tokens, heads,
    extent]`, where key `j` adds `rel[i, h, i - j]` to query `i`'s score when `0 <= i - j < extent`. Returns
    fp32 `(out [tokens, heads, dim], logsumexp [tokens, heads])`.
    """
    seq_len, num_heads, head_dim = q.shape
    group = num_heads // k.shape[1]
    extent = rel.shape[-1]
    keys = k.float().repeat_interleave(group, dim=1).permute(1, 2, 0)
    values = v.float().repeat_interleave(group, dim=1).transpose(0, 1)
    key_idx = torch.arange(seq_len, device=q.device)
    outputs, logsumexps = [], []
    for start in range(0, seq_len, query_chunk):
        query_idx = torch.arange(start, min(start + query_chunk, seq_len), device=q.device)
        distance = query_idx[:, None] - key_idx[None, :]
        visible = (distance >= 0) & (doc_ids[query_idx][:, None] == doc_ids[None, :])
        if window is not None:
            visible = visible & (distance < window)
        index = distance.clamp(0, extent - 1).expand(num_heads, -1, -1)
        bias = rel[query_idx].float().transpose(0, 1).gather(-1, index)
        bias = bias.masked_fill(distance >= extent, 0.0)
        scores = (q[query_idx].float().transpose(0, 1) @ keys) * head_dim**-0.5 + bias
        scores = scores.masked_fill(~visible, float("-inf"))
        logsumexp = torch.logsumexp(scores, dim=-1)
        outputs.append((torch.exp(scores - logsumexp.unsqueeze(-1)) @ values).transpose(0, 1))
        logsumexps.append(logsumexp.transpose(0, 1))
    return torch.cat(outputs), torch.cat(logsumexps)


def flex_relative_attention(
    q: Tensor, k: Tensor, v: Tensor, rel: Tensor, block_mask: BlockMask
) -> tuple[Tensor, Tensor]:
    """`dense_relative_attention` on FlexAttention's fused kernel, in fp32. `block_mask` covers the row padded
    to `flex_length` tokens."""
    seq_len, extent = q.shape[0], rel.shape[-1]
    padding = block_mask.seq_lengths[0] - seq_len
    q, k, v, rel = (F.pad(tensor.transpose(0, 1).unsqueeze(0), (0, 0, 0, padding)) for tensor in (q, k, v, rel))

    def score_mod(score: Tensor, batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        distance = q_idx - kv_idx
        bias = rel[batch, head, q_idx, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, torch.zeros_like(bias))

    out, aux = compiled_flex_attention()(
        q.float(),
        k.float(),
        v.float(),
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
    )
    return out[0, :, :seq_len].transpose(0, 1), aux.lse[0, :, :seq_len].transpose(0, 1)


class Inkling2RMSNorm(nn.Module):
    """RMSNorm computed in fp32 and rounded once to the parameter dtype, whatever the input dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        return rms_norm(hidden_states, self.weight, self.eps, self.weight.dtype)


class Inkling2ShortConv(nn.Module):
    """Causal depthwise convolution over each document's last `kernel_size` tokens plus an identity residual,
    scaled by `(kernel_size + 1) ** -0.5`. `weight[:, 0, -1]` taps the current token."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        kernel_size = self.weight.shape[-1]
        value = x.float()
        taps = self.weight.float()
        out = torch.zeros_like(value)
        for tap in range(kernel_size):
            out = out + lagged(value, kernel_size - 1 - tap, positions) * taps[:, 0, tap]
        return ((out + value) * (kernel_size + 1) ** -0.5).to(x.dtype)


class Inkling2DynamicShortConv(nn.Module):
    """Short convolution of a full-attention layer's keys or values. Each token adds a per-head projection of
    its attention input to the static taps. Scaled by `(2 * kernel_size + 1) ** -0.5`; `static_w[0]` taps the
    current token."""

    def __init__(self, channels: int, num_heads: int, kernel_size: int, hidden_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.static_w = nn.Parameter(torch.empty(kernel_size, channels))
        self.weight_proj = nn.Parameter(torch.empty(num_heads * kernel_size, hidden_size))

    def forward(self, x: Tensor, source: Tensor, positions: Tensor) -> Tensor:
        kernel_size, channels = self.static_w.shape
        dynamic = F.linear(source, self.weight_proj).float().view(-1, self.num_heads, kernel_size)
        dynamic = dynamic.repeat_interleave(channels // self.num_heads, dim=1)
        value = x.float()
        static = self.static_w.float()
        out = torch.zeros_like(value)
        for lag in range(kernel_size):
            out = out + lagged(value, lag, positions) * (static[lag] + dynamic[:, :, lag])
        return ((out + value) * (2 * kernel_size + 1) ** -0.5).to(x.dtype)


class Inkling2Attention(nn.Module):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_local = config.is_local_layer(layer_idx)
        self.num_heads = config.swa_num_attention_heads if self.is_local else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_local else config.head_dim
        self.d_rel = config.d_rel
        self.window = config.sliding_window_size if self.is_local else None
        self.qk_clip_threshold = config.qk_l2_clip_threshold
        self.rel_clip_threshold = config.rel_l2_clip_threshold
        self.value_norm_eps = config.rms_norm_eps if config.use_value_norm_no_gamma else None
        # The query's length scaling only applies where attention spans the whole document.
        self.log_scaling = (
            (config.log_scaling_n_floor, config.log_scaling_alpha)
            if not self.is_local and config.log_scaling_n_floor is not None
            else None
        )

        hidden_size = config.hidden_size
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.rel_proj = nn.Linear(hidden_size, self.num_heads * self.d_rel, bias=False)
        self.gate_proj = nn.Linear(hidden_size, query_dim, bias=False)
        self.sink_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.o_proj = nn.Linear(query_dim, hidden_size, bias=False)
        self.q_scale = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.rel_scale = nn.Parameter(torch.empty(self.num_heads, self.d_rel))
        rel_extent = config.sliding_window_size if self.is_local else config.rel_extent
        self.rel_logits_proj = nn.Parameter(torch.empty(self.d_rel, rel_extent))
        if self.is_local:
            self.k_sconv = Inkling2ShortConv(kv_dim, config.sconv_kernel_size)
            self.v_sconv = Inkling2ShortConv(kv_dim, config.sconv_kernel_size)
        else:
            self.k_sconv = Inkling2DynamicShortConv(kv_dim, self.num_kv_heads, config.sconv_kernel_size, hidden_size)
            self.v_sconv = Inkling2DynamicShortConv(kv_dim, self.num_kv_heads, config.sconv_kernel_size, hidden_size)

    def attention_flops(self, seq_len: int) -> int:
        return quadratic_attention_flops(
            num_heads=self.num_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            seq_len=seq_len,
            attended_tokens=min(seq_len, self.window or seq_len),
        )

    def forward(
        self,
        query_input: Tensor,
        key_input: Tensor,
        value_input: Tensor,
        positions: Tensor,
        doc_ids: Tensor,
        block_mask: BlockMask | None,
    ) -> Tensor:
        """`*_input` are `[tokens, hidden]`. Without a `block_mask` the attention runs densely."""
        seq_len = query_input.shape[0]
        query = self.q_proj(query_input)
        rel = self.rel_proj(query_input)
        gate = self.gate_proj(query_input)
        sink = self.sink_proj(query_input)
        key = self.k_proj(key_input)
        value = self.v_proj(value_input)
        if self.is_local:
            key = self.k_sconv(key, positions)
            value = self.v_sconv(value, positions)
        else:
            key = self.k_sconv(key, key_input, positions)
            value = self.v_sconv(value, value_input, positions)

        tau = None
        if self.log_scaling is not None:
            n_floor, alpha = self.log_scaling
            tau = 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))

        query = l2_clip(query.view(seq_len, self.num_heads, self.head_dim), self.qk_clip_threshold, self.q_scale, tau)
        key = l2_clip(key.view(seq_len, self.num_kv_heads, self.head_dim), self.qk_clip_threshold)
        value = value.view(seq_len, self.num_kv_heads, self.head_dim)
        if self.value_norm_eps is not None:
            value = rms_norm(value, None, self.value_norm_eps, value.dtype)
        rel = l2_clip(rel.view(seq_len, self.num_heads, self.d_rel), self.rel_clip_threshold, self.rel_scale, tau)
        rel_logits = rel @ self.rel_logits_proj

        if block_mask is None:
            out, logsumexp = dense_relative_attention(query, key, value, rel_logits, doc_ids, self.window)
        else:
            out, logsumexp = flex_relative_attention(query, key, value, rel_logits, block_mask)

        out = out * torch.sigmoid(gate.float().view(seq_len, self.num_heads, self.head_dim))
        out = out * torch.sigmoid(logsumexp - sink.float()).unsqueeze(-1)
        return self.o_proj(out.reshape(seq_len, -1).to(query_input.dtype))


class Inkling2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.is_local = config.is_local_layer(layer_idx)
        self.starts_block = layer_idx % config.n_layers_per_attn_res_block == 0
        self.normalize_values = config.attn_res_normalize_values

        # Full-attention layers mix separate inputs for queries, keys and values. The first layer reads the
        # embeddings directly, so it has no attention query.
        if layer_idx == 0:
            self.register_parameter("attn_query", None)
        else:
            self.attn_query = nn.Parameter(torch.empty(hidden_size) if self.is_local else torch.empty(3, hidden_size))
        self.mlp_query = nn.Parameter(torch.empty(hidden_size))

        if self.is_local:
            self.attn_norm = Inkling2RMSNorm(hidden_size, config.rms_norm_eps)
        else:
            self.q_input_norm = Inkling2RMSNorm(hidden_size, config.rms_norm_eps)
            self.k_input_norm = Inkling2RMSNorm(hidden_size, config.rms_norm_eps)
            self.v_input_norm = Inkling2RMSNorm(hidden_size, config.rms_norm_eps)
        self.self_attn = Inkling2Attention(config, layer_idx)
        self.attn_sconv = Inkling2ShortConv(hidden_size, config.sconv_kernel_size)

        self.mlp_norm = Inkling2RMSNorm(hidden_size, config.rms_norm_eps)
        self.mlp = MoE(
            router=TokenChoiceTopKRouter(
                dim=hidden_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                score_func="sigmoid",
                route_norm=True,
                route_scale=config.route_scale,
                selection_bias=config.use_gate_bias,
            ),
            experts=GroupedExperts(
                dim=hidden_size,
                hidden_dim=config.intermediate_size,
                num_experts=config.n_routed_experts,
                activation=SoftcappedSwiglu(config.mlp_gate_softcap, config.mlp_up_softcap),
            ),
            shared_expert=None,
            score_before_experts=False,
            # The selection bias is a trained part of the checkpoint, not a load-balancing state.
            load_balance_coeff=None,
        )
        self.mlp_sconv = Inkling2ShortConv(hidden_size, config.sconv_kernel_size)

    def forward(
        self,
        blocks: tuple[Tensor, ...],
        num_blocks: Tensor,
        partial: Tensor,
        positions: Tensor,
        doc_ids: Tensor,
        routed_experts: Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        """Returns the partial block after this layer. The first `num_blocks` (a scalar tensor) of `blocks` are
        the completed blocks, the normalized embeddings first; all states are `[tokens, hidden]`."""
        if self.attn_query is None:
            streams = [partial] * (1 if self.is_local else 3)
        else:
            streams = attn_res_mix(blocks, num_blocks, [partial], self.attn_query, self.normalize_values)
        if self.is_local:
            query_input = key_input = value_input = self.attn_norm(streams[0])
        else:
            query_input = self.q_input_norm(streams[0])
            key_input = self.k_input_norm(streams[1])
            value_input = self.v_input_norm(streams[2])
        attn_output = self.self_attn(query_input, key_input, value_input, positions, doc_ids, block_mask)
        attn_output = self.attn_sconv(attn_output, positions)

        # A layer that starts a block closes the previous one: `partial` becomes a completed block and the
        # new partial block so far is the attention output.
        tail = [partial, attn_output] if self.starts_block else [partial.float() + attn_output.float()]
        (mlp_input,) = attn_res_mix(blocks, num_blocks, tail, self.mlp_query, self.normalize_values)
        mlp_output = self.mlp(self.mlp_norm(mlp_input).unsqueeze(0), routed_experts=routed_experts).squeeze(0)
        mlp_output = self.mlp_sconv(mlp_output, positions)

        if self.starts_block:
            return attn_output + mlp_output
        return (partial.float() + attn_output.float() + mlp_output.float()).to(partial.dtype)


class Inkling2PreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Inkling2Config
    base_model_prefix = "model"

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        return "the short convolutions and the relative-bias attention are not context-parallel yet"

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(("attn_query", "mlp_query", "output_query", "mlp.router.selection_bias"))

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def _init_weights(self, module: nn.Module) -> None:
        """As in the transformers reference: the mix queries and the dynamic convolutions' static taps start at
        zero, the clip scales at one."""
        super()._init_weights(module)
        init_std = self.config.get_text_config().initializer_range
        if isinstance(module, Inkling2ShortConv):
            nn.init.normal_(module.weight, std=init_std)
        elif isinstance(module, Inkling2DynamicShortConv):
            nn.init.zeros_(module.static_w)
            nn.init.normal_(module.weight_proj, std=init_std)
        elif isinstance(module, Inkling2Attention):
            nn.init.ones_(module.q_scale)
            nn.init.ones_(module.rel_scale)
            nn.init.normal_(module.rel_logits_proj, std=init_std)
        elif isinstance(module, Inkling2DecoderLayer):
            if module.attn_query is not None:
                nn.init.zeros_(module.attn_query)
            nn.init.zeros_(module.mlp_query)
        elif isinstance(module, Inkling2TextModel):
            nn.init.zeros_(module.output_query)


class Inkling2TextModel(Inkling2PreTrainedModel):
    config_class = Inkling2TextConfig

    def __init__(self, config: Inkling2TextConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.empty(config.hidden_size))
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        num_padding_tokens = config.vocab_size - config.unpadded_vocab_size
        if num_padding_tokens > 0:
            self.register_buffer("unembed_padding", torch.zeros(num_padding_tokens, config.hidden_size))
        # `auto` runs FlexAttention on CUDA and the dense reference elsewhere.
        self.attention_backend: AttentionBackend = "auto"
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        """`routed_experts [1, tokens, layers, top_k]` replays the rollout's routing. `seq_lens` are the
        packed row's per-document lengths; positions restart at every document."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if seq_lens_are_pre_shard:
            raise ValueError("Inkling 2 does not support context parallelism")

        # Like serving, the norm also applies to embeddings passed in.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"Inkling 2 takes one packed row per forward, got batch size {inputs_embeds.shape[0]}")
        partial = self.embed_norm(inputs_embeds.squeeze(0))
        seq_len = partial.shape[0]

        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens.to(device=partial.device), total_tokens=seq_len)
        positions, doc_ids = document_layout(cu_seqlens)

        use_flex = self.attention_backend == "flex" or (self.attention_backend == "auto" and partial.is_cuda)
        local_mask = global_mask = None
        if use_flex:
            padded_doc_ids = F.pad(doc_ids, (0, flex_length(seq_len) - seq_len), value=-1)
            local_mask = relative_block_mask(padded_doc_ids, self.config.sliding_window_size)
            global_mask = relative_block_mask(padded_doc_ids, None)

        layers = self.layers[: self.config.num_hidden_layers]
        # One slot per block. A block that is not complete yet is a tensor of zeros of its own, which
        # `num_blocks` keeps out of the mixes: a compiled layer then gets the same inputs at every depth, and
        # each layer type compiles once.
        blocks = [partial * 0 for decoder_layer in layers if decoder_layer.starts_block]
        num_blocks = 0
        for layer_idx, decoder_layer in enumerate(layers):
            # .contiguous(): the per-layer slice is a strided view, and the compiled MoE asserts a contiguous input.
            routed_experts_layer = (
                routed_experts[:, :, layer_idx, :].contiguous() if routed_experts is not None else None
            )
            next_partial = decoder_layer(
                tuple(blocks),
                torch.tensor(num_blocks, device=partial.device),
                partial,
                positions,
                doc_ids,
                routed_experts=routed_experts_layer,
                block_mask=local_mask if decoder_layer.is_local else global_mask,
            )
            if decoder_layer.starts_block:
                blocks[num_blocks] = partial
                num_blocks += 1
            partial = next_partial

        (hidden_states,) = attn_res_mix(
            blocks,
            torch.tensor(num_blocks, device=partial.device),
            [partial],
            self.output_query,
            self.config.attn_res_normalize_values,
        )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states).unsqueeze(0))


class Inkling2Model(nn.Module):
    def __init__(self, config: Inkling2Config) -> None:
        super().__init__()
        self.language_model = Inkling2TextModel(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(embeddings)

    def forward(self, *args, **kwargs) -> BaseModelOutputWithPast:
        return self.language_model(*args, **kwargs)


class Inkling2ForCausalLM(Inkling2PreTrainedModel):
    """Text-only Inkling 2. The vision tower and the speculator are not loaded.

    The LM head covers the real vocabulary only: serving masks the padding rows of the checkpoint's
    unembedding out of the softmax, so token logprobs normalize over `unpadded_vocab_size` tokens.
    """

    def __init__(self, config: Inkling2Config) -> None:
        super().__init__(config)
        text_config = config.text_config
        self.model = Inkling2Model(config)
        self.vocab_size = text_config.unpadded_vocab_size
        self.lm_head = VanillaOutputLinear(
            text_config.hidden_size, text_config.unpadded_vocab_size, softcap=text_config.final_logit_softcapping
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | Tensor = 0,
        temperature: Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        support: TokenSupport | None = None,
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

    def init_buffers_post_meta(self) -> None:
        # The selection biases and the unembedding padding are checkpoint tensors: a load overwrites these
        # zeros, and a model initialized from scratch keeps them.
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()
                if module.router.selection_bias is not None:
                    module.router.selection_bias.zero_()
        unembed_padding = getattr(self.model.language_model, "unembed_padding", None)
        if unembed_padding is not None:
            unembed_padding.zero_()


__all__ = [
    "Inkling2DecoderLayer",
    "Inkling2ForCausalLM",
    "Inkling2Model",
    "Inkling2PreTrainedModel",
    "Inkling2TextModel",
]
