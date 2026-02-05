#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from flash_attn_interface import _flash_attn_backward as flash_attn_3_backward
from flash_attn_interface import _flash_attn_forward as flash_attn_3_forward
from huggingface_hub import hf_hub_download
from ring_flash_attn import llama3_flash_attn_varlen_func, update_ring_flash_attn_params
from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS
from ring_flash_attn.utils import AllGatherComm, get_default_args

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb


def flash_attn_3_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout_p != 0.0:
        raise ValueError("flash-attention-3 benchmark wrapper only supports dropout_p=0.0")

    params = get_default_args(flash_attn_3_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
        }
    )
    outputs = flash_attn_3_forward(**params)
    if len(outputs) != 4:
        raise ValueError(f"Unexpected flash-attention-3 forward output count: {len(outputs)}")
    out, lse, _, _ = outputs
    return out, lse


def flash_attn_3_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> None:
    params = get_default_args(flash_attn_3_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "deterministic": deterministic,
        }
    )
    flash_attn_3_backward(**params)


def llama3_flash_attn_3_varlen_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_k_stride: int,
    local_k_slice: slice,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    out_list = []
    lse_list = []

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    if nheads_k % heads_k_stride != 0:
        raise ValueError("num_key_value_heads must be divisible by heads_k_stride")

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm = AllGatherComm(process_group)

    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]

        block_out, block_lse = flash_attn_3_varlen_forward(
            q=q_i,
            k=k_i,
            v=v_i,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
        )
        out_list.append(block_out)
        lse_list.append(block_lse)

    out = torch.cat(out_list, dim=1)
    lse = torch.cat(lse_list, dim=-2)
    return out, lse


def llama3_flash_attn_3_varlen_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_k_stride: int,
    local_k_slice: slice,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dropout_p != 0.0:
        raise ValueError("flash-attention-3 benchmark wrapper only supports dropout_p=0.0")

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    if nheads_k % heads_k_stride != 0:
        raise ValueError("num_key_value_heads must be divisible by heads_k_stride")

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    dkv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    if heads_k_stride != nheads_k:
        kv_contiguous_buffer = torch.empty(
            (2, total_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = AllGatherComm(process_group)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        dkv_buffer.zero_()

        q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
        q_i = q[:, q_slice]
        dout_i = dout[:, q_slice]
        out_i = out[:, q_slice]
        dq_i = dq[:, q_slice]
        lse_i = softmax_lse[q_slice].contiguous() if softmax_lse.dim() != 3 else softmax_lse[:, q_slice].contiguous()

        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]
        dk_i = dkv_buffer[0][local_k_slice]
        dv_i = dkv_buffer[1][local_k_slice]

        flash_attn_3_varlen_backward(
            dout=dout_i,
            q=q_i,
            k=k_i,
            v=v_i,
            out=out_i,
            softmax_lse=lse_i,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq_i,
            dk=dk_i,
            dv=dv_i,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )

        if heads_k_stride != nheads_k:
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        else:
            dk_i = dk
            dv_i = dv

        dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=process_group)
        dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=process_group)

        if heads_k_stride != nheads_k:
            dk[:, i : i + heads_k_stride] = dk_i
            dv[:, i : i + heads_k_stride] = dv_i

    return dq, dk, dv


class Llama3FlashAttn3VarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        heads_k_stride: int,
        local_k_slice: slice,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        return_softmax: bool,
        group: dist.ProcessGroup,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if alibi_slopes is not None:
            raise ValueError("alibi is not supported in flash-attention-3 benchmark wrapper")

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = llama3_flash_attn_3_varlen_forward(
            process_group=group,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            heads_k_stride=heads_k_stride,
            local_k_slice=local_k_slice,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.heads_k_stride = heads_k_stride
        ctx.local_k_slice = local_k_slice
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.group = group
        if return_softmax:
            return out, softmax_lse, None
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = llama3_flash_attn_3_varlen_backward(
            process_group=ctx.group,
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            heads_k_stride=ctx.heads_k_stride,
            local_k_slice=ctx.local_k_slice,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            deterministic=ctx.deterministic,
        )
        return (dq, dk, dv) + (None,) * 15


def llama3_flash_attn_3_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_k_stride: int,
    local_k_slice: slice,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup | None = None,
):
    if group is None:
        raise ValueError("process group must be provided for ring flash-attention-3")
    return Llama3FlashAttn3VarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


class RingFlashAttention(FlashAttention):
    def __init__(
        self,
        config: AttentionConfig,
        process_group: dist.ProcessGroup,
        heads_k_stride: int,
        use_fa3: bool = False,
    ):
        super().__init__(config, flash_attn_version=2)
        self.process_group = process_group
        self.heads_k_stride = heads_k_stride
        self.use_fa3 = use_fa3

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del cu_seqlens, max_seqlen

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        ring_func = llama3_flash_attn_3_varlen_func if self.use_fa3 else llama3_flash_attn_varlen_func
        out = ring_func(
            query_states.transpose(1, 2)[0],
            key_states.transpose(1, 2)[0],
            value_states.transpose(1, 2)[0],
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            local_k_slice=DATA_PARAMS["local_k_slice"],
            causal=True,
            group=self.process_group,
            heads_k_stride=self.heads_k_stride,
        )

        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FA2 vs FA3 vs ring-flash-attention")
    parser.add_argument("--model-name", default="PrimeIntellect/INTELLECT-3")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument("--dp-tokens-per-rank", type=int, default=65_000)
    parser.add_argument("--ring-global-seq-len", type=int, default=130_000)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--heads-k-stride", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--modes",
        default="fa2,fa3,ring_cp2,ring_cp2_fa3",
        help="Comma-separated subset of: fa2,fa3,ring_cp2,ring_cp2_fa3",
    )
    parser.add_argument(
        "--layouts",
        default="one_long,multi_docs",
        help="Comma-separated subset of: one_long,multi_docs",
    )
    parser.add_argument("--multi-doc-min-len", type=int, default=256)
    parser.add_argument("--multi-doc-max-len", type=int, default=24_000)
    parser.add_argument("--multi-doc-target-len", type=int, default=8_192)
    parser.add_argument("--profile", action="store_true", help="Run torch profiler for each mode")
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--profile-top-k", type=int, default=12)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("benchmarks/results/profiles/flash_attn_variants"),
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/flash_attn_variants_intellect3.json"))
    return parser.parse_args()


def load_intellect3_shape(model_name: str) -> dict[str, int | float | bool | None]:
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with Path(config_path).open() as f:
        cfg = json.load(f)

    return {
        "hidden_size": int(cfg["hidden_size"]),
        "head_dim": int(cfg["head_dim"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "attention_bias": bool(cfg["attention_bias"]),
        "use_qk_norm": bool(cfg["use_qk_norm"]),
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
        "rope_theta": float(cfg["rope_theta"]),
        "partial_rotary_factor": float(cfg.get("partial_rotary_factor", 1.0)),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
    }


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def build_rotary_embeddings(
    *,
    position_ids: torch.Tensor,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = int(round(head_dim * partial_rotary_factor))
    rotary_dim = max(2, min(head_dim, rotary_dim))
    if rotary_dim % 2 == 1:
        rotary_dim -= 1

    positions = position_ids.squeeze(0).to(device=device, dtype=torch.float32)
    inv_idx = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (inv_idx / rotary_dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().unsqueeze(0).to(dtype=dtype)
    sin = emb.sin().unsqueeze(0).to(dtype=dtype)
    return cos, sin


def sample_doc_lengths(
    *,
    total_tokens: int,
    min_len: int,
    max_len: int,
    target_len: int,
    rng: random.Random,
) -> list[int]:
    min_docs = max(1, math.ceil(total_tokens / max_len))
    max_docs = max(1, total_tokens // min_len)
    target_docs = max(min_docs, min(max_docs, round(total_tokens / target_len)))

    if max_docs >= 2:
        low = max(min_docs, int(target_docs * 0.8))
        high = min(max_docs, int(target_docs * 1.2))
        if low > high:
            low = high = target_docs
        num_docs = max(2, rng.randint(low, high))
    else:
        num_docs = 1

    base_total = num_docs * min_len
    if base_total > total_tokens:
        return [total_tokens]

    remaining = total_tokens - base_total
    caps = [max_len - min_len] * num_docs
    extras = [0] * num_docs

    for idx in range(num_docs - 1):
        max_take = min(caps[idx], remaining)
        remaining_capacity = sum(caps[idx + 1 :])
        min_take = max(0, remaining - remaining_capacity)
        if max_take > min_take:
            take = rng.randint(min_take, max_take)
        else:
            take = max_take
        extras[idx] = take
        remaining -= take

    extras[-1] = remaining
    lengths = [min_len + extra for extra in extras]
    return lengths


def build_cu_seqlens(doc_lengths: list[int], device: torch.device) -> torch.Tensor:
    lengths = torch.tensor(doc_lengths, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens


def build_position_ids(doc_lengths: list[int], device: torch.device) -> torch.Tensor:
    parts = [torch.arange(length, device=device, dtype=torch.long) for length in doc_lengths]
    return torch.cat(parts, dim=0).unsqueeze(0)


def summarize_profile(
    profiler: torch.profiler.profile,
    *,
    top_k: int,
) -> dict[str, object]:
    events = profiler.key_averages()
    top_events = sorted(events, key=lambda event: event.self_device_time_total, reverse=True)[:top_k]
    total_self_device_us = sum(event.self_device_time_total for event in events)

    kernels = []
    for event in top_events:
        kernels.append(
            {
                "name": event.key,
                "self_device_time_ms": float(event.self_device_time_total / 1000.0),
                "device_time_total_ms": float(event.device_time_total / 1000.0),
                "self_cpu_time_ms": float(event.self_cpu_time_total / 1000.0),
                "count": int(event.count),
                "self_device_pct": float(100.0 * event.self_device_time_total / total_self_device_us)
                if total_self_device_us > 0
                else 0.0,
            }
        )

    return {
        "total_self_device_time_ms": float(total_self_device_us / 1000.0),
        "top_kernels": kernels,
    }


def benchmark_mode(
    *,
    mode: str,
    layout: str,
    shape: dict[str, int | float | bool | None],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    mode_idx = {"fa2": 1, "fa3": 2, "ring_cp2": 3, "ring_cp2_fa3": 4}[mode]
    layout_idx = {"one_long": 1, "multi_docs": 2}[layout]
    base_seed = args.seed + 10_000 * mode_idx + 1_000_000 * layout_idx
    local_rng = random.Random(base_seed + 101 * rank)

    if mode in {"fa2", "fa3"}:
        local_seq_len = args.dp_tokens_per_rank
        global_tokens_per_step = args.dp_tokens_per_rank * world_size
        if layout == "one_long":
            local_doc_lengths = [local_seq_len]
        elif layout == "multi_docs":
            local_doc_lengths = sample_doc_lengths(
                total_tokens=local_seq_len,
                min_len=args.multi_doc_min_len,
                max_len=args.multi_doc_max_len,
                target_len=args.multi_doc_target_len,
                rng=local_rng,
            )
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        position_ids = build_position_ids(local_doc_lengths, device=device)
        cu_seqlens = build_cu_seqlens(local_doc_lengths, device=device)
        max_seqlen = int(max(local_doc_lengths))
        doc_scope = "local_per_rank"
        doc_lengths_for_stats = local_doc_lengths

    elif mode in {"ring_cp2", "ring_cp2_fa3"}:
        if world_size != args.cp_size:
            raise ValueError(f"ring_cp2 requires world_size == cp_size ({args.cp_size}), got {world_size}")
        if args.ring_global_seq_len % args.cp_size != 0:
            raise ValueError("ring_global_seq_len must be divisible by cp_size")

        local_seq_len = args.ring_global_seq_len // args.cp_size
        global_tokens_per_step = args.ring_global_seq_len

        global_doc_lengths_payload: list[list[int] | None] = [None]
        if rank == 0:
            if layout == "one_long":
                global_doc_lengths_payload[0] = [args.ring_global_seq_len]
            elif layout == "multi_docs":
                global_rng = random.Random(base_seed + 17)
                global_doc_lengths_payload[0] = sample_doc_lengths(
                    total_tokens=args.ring_global_seq_len,
                    min_len=args.multi_doc_min_len,
                    max_len=args.multi_doc_max_len,
                    target_len=args.multi_doc_target_len,
                    rng=global_rng,
                )
            else:
                raise ValueError(f"Unsupported layout: {layout}")

        dist.broadcast_object_list(global_doc_lengths_payload, src=0, group=process_group)
        global_doc_lengths = global_doc_lengths_payload[0]
        if global_doc_lengths is None:
            raise ValueError("Failed to create global doc lengths")

        global_cu_seqlens = build_cu_seqlens(global_doc_lengths, device=device)
        update_ring_flash_attn_params(global_cu_seqlens, process_group)

        global_position_ids = build_position_ids(global_doc_lengths, device=device)
        local_start = rank * local_seq_len
        local_end = local_start + local_seq_len
        position_ids = global_position_ids[:, local_start:local_end].contiguous()

        cu_seqlens = None
        max_seqlen = None
        doc_scope = "global"
        doc_lengths_for_stats = global_doc_lengths
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    attn_config = AttentionConfig(
        hidden_size=int(shape["hidden_size"]),
        head_dim=int(shape["head_dim"]),
        num_attention_heads=int(shape["num_attention_heads"]),
        num_key_value_heads=int(shape["num_key_value_heads"]),
        is_causal=True,
        attention_bias=bool(shape["attention_bias"]),
        use_qk_norm=bool(shape["use_qk_norm"]),
        rms_norm_eps=float(shape["rms_norm_eps"]),
    )

    if mode == "fa2":
        layer: torch.nn.Module = FlashAttention(attn_config, flash_attn_version=2)
    elif mode == "fa3":
        layer = FlashAttention(attn_config, flash_attn_version=3)
    elif mode == "ring_cp2":
        layer = RingFlashAttention(attn_config, process_group=process_group, heads_k_stride=args.heads_k_stride)
    elif mode == "ring_cp2_fa3":
        layer = RingFlashAttention(
            attn_config,
            process_group=process_group,
            heads_k_stride=args.heads_k_stride,
            use_fa3=True,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    layer = layer.to(device=device, dtype=dtype)
    hidden_states = torch.randn(1, local_seq_len, int(shape["hidden_size"]), device=device, dtype=dtype)
    position_embeddings = build_rotary_embeddings(
        position_ids=position_ids,
        head_dim=int(shape["head_dim"]),
        partial_rotary_factor=float(shape["partial_rotary_factor"]),
        rope_theta=float(shape["rope_theta"]),
        dtype=dtype,
        device=device,
    )

    step_times = []
    total_steps = args.warmup_steps + args.bench_steps

    dist.barrier(group=process_group)
    for step in range(total_steps):
        if step == args.warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)

        dist.barrier(group=process_group)
        torch.cuda.synchronize(device)
        start_t = time.perf_counter()

        layer.zero_grad(set_to_none=True)
        out, _ = layer(hidden_states, position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        loss = out.float().square().mean()
        loss.backward()

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_t
        elapsed_tensor = torch.tensor(elapsed, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=process_group)

        if step >= args.warmup_steps:
            step_times.append(float(elapsed_tensor.item()))

    peak_mem_tensor = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
    dist.all_reduce(peak_mem_tensor, op=dist.ReduceOp.MAX, group=process_group)

    profile_summary = None
    if args.profile:
        profile_dir = args.profile_dir / layout / mode
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / f"rank{rank}.trace.json"
        table_path = profile_dir / f"rank{rank}.table.txt"

        dist.barrier(group=process_group)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(args.profile_steps):
                dist.barrier(group=process_group)
                layer.zero_grad(set_to_none=True)
                out, _ = layer(hidden_states, position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                loss = out.float().square().mean()
                loss.backward()
                torch.cuda.synchronize(device)
                profiler.step()

        profiler.export_chrome_trace(str(trace_path))
        table_path.write_text(
            profiler.key_averages().table(sort_by="self_device_time_total", row_limit=args.profile_top_k)
        )
        profile_summary = summarize_profile(profiler, top_k=args.profile_top_k)
        profile_summary["trace_path"] = str(trace_path)
        profile_summary["table_path"] = str(table_path)
        profile_summary["rank"] = rank
        dist.barrier(group=process_group)

    del out, loss, layer, hidden_states, position_embeddings
    torch.cuda.empty_cache()

    throughput = [global_tokens_per_step / t for t in step_times]
    result = {
        "layout": layout,
        "mode": mode,
        "global_tokens_per_step": int(global_tokens_per_step),
        "local_seq_len": int(local_seq_len),
        "world_size": int(world_size),
        "doc_scope": doc_scope,
        "doc_count": int(len(doc_lengths_for_stats)),
        "min_doc_len": int(min(doc_lengths_for_stats)),
        "max_doc_len": int(max(doc_lengths_for_stats)),
        "step_time_s": summary(step_times),
        "throughput_tokens_per_s": summary(throughput),
        "peak_memory_gib": float(peak_mem_tensor.item() / (1024**3)),
    }
    if profile_summary is not None:
        result["profile"] = profile_summary
    return result


def format_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| Layout | Mode | Docs | Min Doc Len | Max Doc Len | Global Tokens/Step | Local Seq/Rank | Step Time Mean (s) | Throughput Mean (tok/s) | Peak Mem (GiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        step_time = result["step_time_s"]
        throughput = result["throughput_tokens_per_s"]
        lines.append(
            "| {layout} | {mode} | {doc_count} ({doc_scope}) | {min_doc_len} | {max_doc_len} | {global_tokens_per_step} | {local_seq_len} | {step_mean:.4f} | {throughput_mean:.1f} | {peak_memory_gib:.2f} |".format(
                layout=result["layout"],
                mode=result["mode"],
                doc_count=result["doc_count"],
                doc_scope=result["doc_scope"],
                min_doc_len=result["min_doc_len"],
                max_doc_len=result["max_doc_len"],
                global_tokens_per_step=result["global_tokens_per_step"],
                local_seq_len=result["local_seq_len"],
                step_mean=step_time["mean"],
                throughput_mean=throughput["mean"],
                peak_memory_gib=result["peak_memory_gib"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group("nccl", device_id=device)
    process_group = dist.group.WORLD
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    shape_payload: list[dict[str, int | float | bool | None] | None] = [None]
    if rank == 0:
        shape_payload[0] = load_intellect3_shape(args.model_name)
    dist.broadcast_object_list(shape_payload, src=0, group=process_group)
    shape = shape_payload[0]
    if shape is None:
        raise ValueError("Failed to load model shape")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"fa2", "fa3", "ring_cp2", "ring_cp2_fa3"}
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {sorted(valid_modes)}")

    layouts = [layout.strip() for layout in args.layouts.split(",") if layout.strip()]
    valid_layouts = {"one_long", "multi_docs"}
    for layout in layouts:
        if layout not in valid_layouts:
            raise ValueError(f"Invalid layout '{layout}'. Valid layouts: {sorted(valid_layouts)}")

    if args.multi_doc_min_len <= 0:
        raise ValueError("multi-doc-min-len must be positive")
    if args.multi_doc_max_len < args.multi_doc_min_len:
        raise ValueError("multi-doc-max-len must be >= multi-doc-min-len")
    if args.multi_doc_target_len < args.multi_doc_min_len or args.multi_doc_target_len > args.multi_doc_max_len:
        raise ValueError("multi-doc-target-len must be in [multi-doc-min-len, multi-doc-max-len]")

    results = []
    for layout in layouts:
        for mode in modes:
            result = benchmark_mode(
                mode=mode,
                layout=layout,
                shape=shape,
                args=args,
                dtype=dtype,
                device=device,
                process_group=process_group,
                rank=rank,
                world_size=world_size,
            )
            results.append(result)

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": args.model_name,
            "model_shape": shape,
            "benchmark": {
                "warmup_steps": args.warmup_steps,
                "bench_steps": args.bench_steps,
                "dp_tokens_per_rank": args.dp_tokens_per_rank,
                "ring_global_seq_len": args.ring_global_seq_len,
                "cp_size": args.cp_size,
                "dtype": args.dtype,
                "world_size": world_size,
                "layouts": layouts,
                "multi_doc_min_len": args.multi_doc_min_len,
                "multi_doc_max_len": args.multi_doc_max_len,
                "multi_doc_target_len": args.multi_doc_target_len,
                "profile": args.profile,
                "profile_steps": args.profile_steps,
                "profile_top_k": args.profile_top_k,
            },
            "results": results,
        }
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote benchmark JSON to {args.output}")
        print(format_table(results))

    dist.barrier(group=process_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
