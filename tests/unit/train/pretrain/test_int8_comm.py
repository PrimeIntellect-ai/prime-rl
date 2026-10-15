import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.trainer.pretrain.int8_comm import Int8AllGather, Int8ReduceScatter, dequantize, quantize

# Per-rank chunk: an expert weight shard (quantized), a norm weight (raw), another expert shard (quantized).
SEGMENTS = [(0, 512, True), (512, 6, False), (518, 256, True)]
CHUNK = 774


def reference_values(rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(rank)
    return torch.randn(2, CHUNK, generator=generator) * torch.logspace(-3, 1, CHUNK)


def fake_quantize(chunk: torch.Tensor) -> torch.Tensor:
    out = chunk.clone()
    for offset, numel, quantized in SEGMENTS:
        if quantized:
            out[..., offset : offset + numel] = dequantize(*quantize(chunk[..., offset : offset + numel]))
    return out


def run(rank: int, world_size: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29530 + world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # All-gather: every rank gets every chunk with the expert segments int8-rounded, the rest exact.
    shard = reference_values(rank)[0].to(torch.bfloat16)
    output = torch.empty(world_size * CHUNK, dtype=torch.bfloat16)
    output.view(world_size, CHUNK)[rank] = shard
    Int8AllGather(SEGMENTS)(output, output.view(world_size, CHUNK)[rank], group)
    expected = torch.stack(
        [fake_quantize(reference_values(r)[0].to(torch.bfloat16).float()) for r in range(world_size)]
    )
    torch.testing.assert_close(output.view(world_size, CHUNK).float(), expected.bfloat16().float(), rtol=0, atol=0)

    # Reduce-scatter: rank r gets the fp32 sum of every rank's int8-rounded chunk r.
    grads = torch.cat([reference_values(rank * world_size + r)[1] for r in range(world_size)])
    output = torch.empty(CHUNK)
    Int8ReduceScatter(SEGMENTS)(output, grads, group, dist.ReduceOp.SUM)
    expected = sum(fake_quantize(reference_values(r * world_size + rank)[1]) for r in range(world_size))
    torch.testing.assert_close(output, expected)
    dist.destroy_process_group()


def test_quantize_error_is_within_half_a_step():
    x = torch.randn(4, 1024) * torch.logspace(-4, 2, 1024)
    q, scale = quantize(x)
    step = scale.repeat_interleave(128, dim=-1)
    assert ((dequantize(q, scale) - x).abs() <= step / 2 * (1 + 1e-6)).all()


@pytest.mark.parametrize("world_size", [2, 3])
def test_int8_collectives_match_fake_quantization(world_size):
    mp.spawn(run, args=(world_size,), nprocs=world_size)
