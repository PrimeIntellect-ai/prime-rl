"""FLA's CP custom ops match ``build_cp_context`` plus FLA's public CP functions bitwise, on 2 and 4 GPUs.

Each case shards one packed row over the CP ranks and compares outputs and every input gradient with ``torch.equal``,
eager, under ``torch.compile(fullgraph=True)``, inside activation checkpointing, and both together.

    torchrun --nproc-per-node=2 tests/unit/train/models/test_fla_ops_cp.py
"""

import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.modules.causal_conv1d import causal_conv1d as fla_causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from fla.ops.kda import chunk_kda as fla_chunk_kda
from torch.utils.checkpoint import checkpoint

from prime_rl.trainer.models.layers import fla_ops
from prime_rl.utils.cp import CPContext

pytestmark = [pytest.mark.gpu]

TOKENS = 1024
HEADS, HEAD_DIM, CONV_WIDTH, LOWER_BOUND = 4, 128, 4, -5.0


def many_short_documents() -> list[int]:
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 40, (200,), generator=generator).tolist()
    cu_seqlens = [0]
    for length in lengths:
        if cu_seqlens[-1] + length >= TOKENS:
            break
        cu_seqlens.append(cu_seqlens[-1] + length)
    return [*cu_seqlens, TOKENS]


CASES = {
    2: {
        "cut_mid_document": [0, 300, 700, TOKENS],
        "cut_at_rank_boundary": [0, 512, TOKENS],
        "fewer_than_width_tokens_before_cut": [0, 510, TOKENS],
        "one_document_spans_every_rank": [0, TOKENS],
        "rank_with_one_local_document": [0, 100, TOKENS],
        "many_short_documents": many_short_documents(),
    },
    4: {
        "cut_mid_document": [0, 300, 700, TOKENS],
        "cut_at_rank_boundary": [0, 256, 768, TOKENS],
        "fewer_than_width_tokens_before_cut": [0, 254, 513, TOKENS],
        "one_document_spans_every_rank": [0, TOKENS],
        "document_spans_middle_ranks": [0, 100, 900, TOKENS],
        "many_short_documents": many_short_documents(),
    },
}


def local_leaves(rank: int, world_size: int, *tensors: torch.Tensor, sequence: tuple[bool, ...]) -> list:
    """Each rank's leaf: the rank's contiguous slice of a sequence tensor, or the whole of a parameter."""
    local_tokens = TOKENS // world_size
    return [
        (tensor[:, rank * local_tokens : (rank + 1) * local_tokens] if is_sequence else tensor)
        .detach()
        .clone()
        .requires_grad_()
        for tensor, is_sequence in zip(tensors, sequence, strict=True)
    ]


def make_inputs(op: str, rank: int, world_size: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    qkv = (1, TOKENS, HEADS, HEAD_DIM)
    if op == "conv":
        channels = 3 * HEADS * HEAD_DIM
        tensors, sequence = (randn(1, TOKENS, channels), randn(channels, CONV_WIDTH)), (True, False)
        grad_out = randn(1, TOKENS, channels)
    elif op == "gdn":
        tensors = (
            randn(*qkv),
            randn(*qkv),
            randn(*qkv),
            F.logsigmoid(randn(1, TOKENS, HEADS, dtype=torch.float32)),
            randn(1, TOKENS, HEADS).sigmoid(),
        )
        sequence, grad_out = (True,) * 5, randn(*qkv)
    else:
        tensors = (
            randn(*qkv),
            randn(*qkv),
            randn(*qkv),
            randn(*qkv),
            randn(1, TOKENS, HEADS),
            torch.rand(HEADS, device="cuda", generator=generator),
            randn(HEADS * HEAD_DIM, dtype=torch.float32),
        )
        sequence, grad_out = (True,) * 5 + (False,) * 2, randn(*qkv)
    local_tokens = TOKENS // world_size
    return local_leaves(rank, world_size, *tensors, sequence=sequence), grad_out[
        :, rank * local_tokens : (rank + 1) * local_tokens
    ].contiguous()


def reference_fn(op: str, fla_cp_context):
    if op == "conv":
        return lambda x, weight: fla_causal_conv1d(
            x=x, weight=weight, bias=None, activation="silu", cp_context=fla_cp_context
        )[0]
    if op == "gdn":
        return lambda q, k, v, g, beta: fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=fla_cp_context.cu_seqlens,
            cp_context=fla_cp_context,
        )[0]
    return lambda q, k, v, g, beta, A_log, dt_bias: fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=fla_cp_context.cu_seqlens,
        cp_context=fla_cp_context,
    )[0]


def custom_op_fn(op: str, varlen: fla_ops.FlaVarlen, cp_context: CPContext):
    if op == "conv":
        return lambda x, weight: fla_ops.causal_conv1d(x, weight, "silu", varlen, cp_context)
    if op == "gdn":
        return lambda q, k, v, g, beta: fla_ops.chunk_gated_delta_rule(q, k, v, g, beta, varlen, cp_context)
    return lambda q, k, v, g, beta, A_log, dt_bias: fla_ops.chunk_kda(
        q, k, v, g, beta, A_log, dt_bias, LOWER_BOUND, varlen, cp_context
    )


def with_mode(fn, mode: str):
    if mode == "eager":
        return fn
    if mode == "compile":
        return torch.compile(fn, fullgraph=True)
    checkpointed = lambda *inputs: checkpoint(fn, *inputs, use_reentrant=False)  # noqa: E731
    return checkpointed if mode == "checkpoint" else torch.compile(checkpointed, fullgraph=True)


def run(fn, inputs: list[torch.Tensor], grad_out: torch.Tensor) -> list[torch.Tensor]:
    out = fn(*inputs)
    out.backward(grad_out)
    return [out.detach(), *(tensor.grad for tensor in inputs)]


def main() -> None:
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(world_size)))
    cp_context = CPContext(group, rank, world_size)
    failures = 0
    for case, boundaries in CASES[world_size].items():
        cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        varlen = fla_ops.prepare_fla_varlen(cu_seqlens, cp_context)
        fla_cp_context = build_cp_context(cu_seqlens, group=group, conv1d_kernel_size=CONV_WIDTH)
        for op in ("conv", "gdn", "kda"):
            inputs, grad_out = make_inputs(op, rank, world_size)
            reference = run(reference_fn(op, fla_cp_context), inputs, grad_out)
            for mode in ("eager", "compile", "checkpoint", "compile_checkpoint"):
                torch._dynamo.reset()
                ours = run(
                    with_mode(custom_op_fn(op, varlen, cp_context), mode),
                    [t.detach().clone().requires_grad_() for t in inputs],
                    grad_out,
                )
                names = ["out", *(f"grad{index}" for index in range(len(inputs)))]
                mismatched = [
                    f"{name} max abs diff {(a.float() - b.float()).abs().max().item():.3e}"
                    for name, a, b in zip(names, reference, ours, strict=True)
                    if not torch.equal(a, b)
                ]
                failures += bool(mismatched)
                status = "FAIL " + "; ".join(mismatched) if mismatched else "PASS"
                print(f"cp={world_size} rank={rank} {case} {op} {mode}: {status}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_cp_ops_match_fla_bitwise(world_size: int):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world_size}", __file__]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-20000:] + result.stderr[-20000:]


if __name__ == "__main__":
    main()
