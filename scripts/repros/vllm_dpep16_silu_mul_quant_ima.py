#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

INT32_MAX = 2**31 - 1


@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor_int64_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M: tl.int64,
    N: tl.int64,
    y_s_col_stride: tl.int64,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = (pid_m * BLOCK_M).to(tl.int64)
    n_offset = (pid_n * BLOCK_N).to(tl.int64)
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset
    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _warmup_m(args: argparse.Namespace) -> int:
    if args.shape == "minimal-overflow":
        return _round_up(INT32_MAX // args.activation_width + 2, args.alignment)

    max_tokens = args.tokens_per_rank * args.dpep_size
    return _round_up(max_tokens * args.top_k + args.local_experts * (args.alignment - 1), args.alignment)


def _rank_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _default_master_addr() -> str:
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list:
        output = subprocess.check_output(["scontrol", "show", "hostnames", node_list], text=True)
        return output.splitlines()[0]
    return "127.0.0.1"


def _init_distributed(enabled: bool) -> tuple[int, int, int]:
    rank, world_size, local_rank = _rank_env()
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    if enabled and world_size > 1:
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("MASTER_ADDR", _default_master_addr())
        os.environ.setdefault("MASTER_PORT", "29577")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.barrier()
    return rank, world_size, local_rank


def _install_int64_patch():
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    def silu_mul_per_token_group_quant_fp8_colmajor_int64(
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        use_ue8m0: bool | None = None,
        eps: float = 1e-10,
    ):
        group_size = 128
        assert input.ndim == 2
        if output is not None:
            assert output.ndim == 2
        assert input.size(0) % group_size == 0
        assert input.size(1) % (group_size * 2) == 0

        if use_ue8m0 is None:
            use_ue8m0 = is_deep_gemm_e8m0_used()

        M, N = input.size()
        N_2 = N // 2

        fp8_dtype = current_platform.fp8_dtype()
        if output is None:
            output = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

        output_scales = torch.empty(
            ((N_2 // group_size), M), dtype=torch.float32, device=input.device
        ).transpose(0, 1)

        block_m = 8
        block_n = group_size
        assert M % block_m == 0
        assert N_2 % block_n == 0

        finfo = torch.finfo(fp8_dtype)
        fp8_min = -224.0 if current_platform.is_fp8_fnuz() else finfo.min
        fp8_max = 224.0 if current_platform.is_fp8_fnuz() else finfo.max

        grid = (M // block_m, N_2 // block_n)
        _silu_mul_per_token_group_quant_fp8_colmajor_int64_kernel[grid](
            input,
            output,
            output_scales,
            M,
            N,
            output_scales.stride(-1),
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            group_size,
            block_m,
            block_n,
        )

        return output, output_scales

    fp8_utils.silu_mul_per_token_group_quant_fp8_colmajor = silu_mul_per_token_group_quant_fp8_colmajor_int64
    return silu_mul_per_token_group_quant_fp8_colmajor_int64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Reproduce vLLM DeepGEMM SiLU/mul FP8 quant int32 address overflow under a DPEP16-style launch.")
    )
    parser.add_argument("--shape", choices=["dpep16-warmup", "minimal-overflow"], default="dpep16-warmup")
    parser.add_argument("--tokens-per-rank", type=int, default=36_000)
    parser.add_argument("--dpep-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--activation-width", type=int, default=4096)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument("--apply-fix", action="store_true")
    parser.add_argument("--no-init-dist", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "1")
    os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    args = _parse_args()
    rank, world_size, local_rank = _init_distributed(not args.no_init_dist)

    if args.apply_fix:
        quant_fn = _install_int64_patch()
    else:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            silu_mul_per_token_group_quant_fp8_colmajor as quant_fn,
        )

    M = _warmup_m(args)
    N = args.activation_width
    assert M % 8 == 0
    assert M % args.alignment == 0
    assert N % (args.alignment * 2) == 0

    input_max_offset = (M - 1) * N + (N - 1)
    output_max_offset = (M - 1) * (N // 2) + ((N // 2) - 1)
    input_gib = M * N * torch.empty((), dtype=torch.bfloat16).element_size() / 1024**3
    output_gib = M * (N // 2) * torch.empty((), dtype=torch.float8_e4m3fn).element_size() / 1024**3

    print(
        f"[rank {rank}/{world_size} host={socket.gethostname()} local_rank={local_rank}] "
        f"shape={args.shape} M={M} N={N} input_max_offset={input_max_offset} "
        f"output_max_offset={output_max_offset} int32_max={INT32_MAX} "
        f"input_gib={input_gib:.2f} output_gib={output_gib:.2f} apply_fix={args.apply_fix}",
        flush=True,
    )

    x = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    y, scales = quant_fn(x, use_ue8m0=False)
    torch.cuda.synchronize()

    print(
        f"[rank {rank}/{world_size}] quant ok output_shape={tuple(y.shape)} scales_shape={tuple(scales.shape)}",
        flush=True,
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
