import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.pretrain.int8_comm import BLOCK, Int8AllGather, Int8ReduceScatter

pytestmark = [pytest.mark.gpu]

# Per-rank chunk: an expert weight shard (quantized), a norm weight (raw), another expert shard (quantized).
SEGMENTS = [(0, 4096, True), (4096, 6, False), (4102, 1024, True)]
CHUNK = 5126


def quantize_reference(x: torch.Tensor) -> torch.Tensor:
    """Blockwise int8 round trip: absmax / 127 scale per 128 values, round half to even."""
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    scale = blocks.abs().amax(-1, keepdim=True) / 127
    q = torch.where(scale > 0, blocks / scale, 0).round()
    return (q * scale).flatten(-2)


def round_trip(chunk: torch.Tensor) -> torch.Tensor:
    out = chunk.float().clone()
    for offset, numel, quantized in SEGMENTS:
        if quantized:
            out[..., offset : offset + numel] = quantize_reference(chunk[..., offset : offset + numel])
    return out


def values(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(CHUNK, generator=generator) * torch.logspace(-3, 1, CHUNK)
    x[:BLOCK] = 0  # an all-zero block (FSDP padding)
    return x.cuda()


def run(rank: int, world_size: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29530 + world_size))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # All-gather: every rank gets every chunk with the expert segments int8-rounded, the rest exact.
    output = torch.empty(world_size, CHUNK, dtype=torch.bfloat16, device="cuda")
    output[rank] = values(rank)
    Int8AllGather(SEGMENTS)(output.view(-1), output[rank], group)
    expected = torch.stack([round_trip(values(r).bfloat16()) for r in range(world_size)]).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    # Reduce-scatter: rank r gets the fp32 sum of every rank's int8-rounded chunk r.
    grads = torch.cat([values(100 + rank * world_size + r) for r in range(world_size)])
    output = torch.empty(CHUNK, device="cuda")
    Int8ReduceScatter(SEGMENTS)(output, grads, group, dist.ReduceOp.SUM)
    expected = sum(round_trip(values(100 + r * world_size + rank)) for r in range(world_size))
    torch.testing.assert_close(output, expected)
    dist.destroy_process_group()


def test_int8_collectives_match_reference_round_trip():
    mp.spawn(run, args=(2,), nprocs=2)
