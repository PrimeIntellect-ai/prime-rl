import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v4.rotary import apply_rotary_pos_emb_interleaved
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear, the first half of the output projection.

    The stacked attention output is `num_attention_heads * head_dim` wide (32768 for
    V4-Flash), so a direct projection to `hidden_size` would dominate the per-token cost.
    Instead the heads are split into `n_groups` groups, each projected independently to
    `out_features / n_groups` channels; a single follow-up linear (`o_b_proj`) mixes the
    concatenation back to `hidden_size`.

    Input is `[..., n_groups, in_features_per_group]`, output `[..., n_groups, out_features / n_groups]`.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


def eager_attention_with_sinks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Eager attention with a per-head learnable sink logit, as in GPT-OSS.

    The sink is one extra logit column per head, concatenated before the softmax and
    dropped from the probabilities afterwards. It absorbs attention mass without
    contributing to the output, so a query can attend to "nothing" rather than being
    forced to spread a full unit of probability over its window.

    `key` and `value` carry a single KV head (`[batch, 1, seq, head_dim]`) which matmul
    broadcasts over every query head; materializing the broadcast would cost
    `num_attention_heads` times the memory for no numerical gain.

    Returns the attention output in `[batch, seq, heads, head_dim]` layout.
    """
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sink_logits = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sink_logits], dim=-1)
    # Row-max subtraction is not free here: without it the exponentials overflow in bf16.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

    scores = F.dropout(probs[..., :-1], p=dropout, training=training).to(value.dtype)
    attn_output = torch.matmul(scores, value)
    return attn_output.transpose(1, 2).contiguous()


def build_sliding_window_mask(
    seq_len: int, sliding_window: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive `[1, 1, seq_len, seq_len]` mask: causal, restricted to a local window.

    Query `q` may attend to key `k` when `k <= q` and `q - k < sliding_window`; every
    other entry is the dtype's minimum, so it vanishes under the softmax.

    Assumes a single document per row. Packed multi-document batches additionally need
    the window clipped at document boundaries; see TODO.md.
    """
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


class DeepseekV4Attention(nn.Module):
    """DeepSeek-V4 self-attention.

    Four things set it apart from a standard attention block:

    1. Shared-KV multi-query attention. `kv_proj` emits a single `head_dim`-wide vector
       per token that serves as both key and value for every query head.
    2. Partial interleaved RoPE on the trailing `qk_rope_head_dim` channels of each head.
       Because the value carries that rotation too, the conjugate rotation is applied to
       the attention output, which leaves each key's contribution a function of its
       relative distance to the query.
    3. A per-head learnable attention sink.
    4. A grouped low-rank output projection (`o_a_proj` then `o_b_proj`).

    Only `sliding_attention` layers are supported so far: `compressor` is always `None`.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type != "sliding_attention":
            raise NotImplementedError(f"DeepSeek-V4 {self.layer_type} layers are not ported yet.")
        # Rope types are labelled `main` / `compress`, independently of `layer_types`:
        # sliding layers take the plain base, the compressed variants share their
        # compressor's base.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        # The CSA / HCA compressors are a later step; sliding layers never have one.
        self.compressor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(self.q_b_norm(q), cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb_interleaved(kv, cos, sin)

        if attention_mask is None:
            attention_mask = build_sliding_window_mask(kv.shape[2], self.sliding_window, kv.dtype, kv.device)

        attn_output = eager_attention_with_sinks(
            q,
            kv,
            kv,
            self.sinks,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )

        # The value stream is the key stream, so it arrived rotated. Rotating the output
        # by the conjugate angle at the query position cancels that out.
        attn_output = apply_rotary_pos_emb_interleaved(attn_output, cos, -sin, unsqueeze_dim=2)

        grouped = self.o_a_proj(attn_output.reshape(*input_shape, self.config.o_groups, -1)).flatten(2)
        return self.o_b_proj(grouped), None

    def init_weights(self, init_std: float) -> None:
        # `init_std` is unused: the sinks are the only parameter this owns outright and
        # they start at zero. Signature kept uniform with the other V4 submodules.
        nn.init.zeros_(self.sinks)


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4GroupedLinear",
    "build_sliding_window_mask",
    "eager_attention_with_sinks",
]
