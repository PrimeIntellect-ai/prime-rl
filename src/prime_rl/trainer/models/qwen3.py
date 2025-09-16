# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Int
from torch import nn

from prime_rl.trainer.models.shared.attention import build_attention
from prime_rl.trainer.models.shared.utils import StateDictAdapter


@dataclass
class Qwen3ModelArgs:
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 151645

    enable_weight_tying: bool = False

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters()) for m in model.children() if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        if self.enable_weight_tying:
            # exclude model.token_embedding parameters from nparams
            nparams = nparams - nparams_embedding

        return nparams, num_flops_per_token


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 1_000_000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

        # RMSNorm added here to the here to include the q-k norm
        # This is one of the main differences between Llama3 and Qwen3
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Adding the q_norm and k_norm here
        # Last layer of adding q-k norm
        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Hidden dimension is directly added from the model argsS
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(dim=model_args.dim, hidden_dim=model_args.hidden_dim)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), rope_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Qwen3ModelBase(nn.Module):
    """
    Qwen3Model Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer("rope_cache", self._precompute_rope_cache(), persistent=False)

        self.layers = nn.ModuleList(
            [TransformerBlock(layer_idx, model_args) for layer_idx in range(model_args.n_layers)]
        )
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        position_ids: Int[torch.Tensor, "batch seq"] | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        # if position_ids is not None:
        # raise NotImplementedError("Position IDs are not supported for Qwen3")

        h = self.tok_embeddings(input_ids)

        for layer in self.layers:
            h = layer(h, self.rope_cache)

        h = self.norm(h) if self.norm else h
        return h


class LogitsWrapper:
    """
    Dummy class to have the same interface as hf code for logits output
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class Qwen3StateDictAdapter(StateDictAdapter):
    """
    This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

    We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
    This can enable us to do a parity test with the HF implementation and make sure that our results are
    aligned with the HF implementation.

    """

    def __init__(self, model_args: Qwen3ModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # todo(sami): check the output with the new format, specifically the layers and lm head that change on our side
        self.from_hf_map = {
            "model.embed_tokens.weight": "model.tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "model.layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "model.layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "model.layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "model.layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "model.layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "model.layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "model.layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "model.layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "model.layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "model.layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "model.layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict


class Qwen3Model(nn.Module):
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.model = Qwen3ModelBase(model_args)
        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

        self.state_dict_adapter = Qwen3StateDictAdapter(model_args, None)

    def init_weights(self):
        self.model.init_weights()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        nn.init.trunc_normal_(
            self.lm_head.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(
        self, input_ids: Int[torch.Tensor, "batch seq"], position_ids: Int[torch.Tensor, "batch seq"] | None = None
    ) -> LogitsWrapper:
        return LogitsWrapper(self.lm_head(self.model(input_ids, position_ids)))

    def state_dict_hf(self):
        return self.state_dict_adapter.to_hf(self.state_dict())

    def load_state_dict_hf(self, state_dict: dict[str, Any]):
        self.load_state_dict(self.state_dict_adapter.from_hf(state_dict))


qwen3_configs = {
    "Qwen/Qwen3-0.6B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "Qwen/Qwen3-1.7B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "Qwen/Qwen3-4B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=9728,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "Qwen/Qwen3-8B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=1000000,
    ),
    "Qwen/Qwen3-14B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=17408,
        rope_theta=1000000,
    ),
    "Qwen/Qwen3-32B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=25600,
        rope_theta=1000000,
    ),
}
