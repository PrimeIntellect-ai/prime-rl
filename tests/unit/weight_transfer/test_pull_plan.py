"""End-to-end CPU tests of the sharded NIXL pull plan: lazy bake -> chain
resolution -> shard routing -> run matching, with the RDMA reads simulated
via ctypes against per-rank shard buffers.

The loaders below mimic the vLLM patterns the bake must survive: fused-QKV
destination narrows, FusedMoE per-expert routing with EP skips, TP source
narrows, padded-vocab partial destinations. The trainer side is modelled as
dim-0 shards split across two ranks, so routing must split each region across
the owning shards.
"""

import ctypes

import pytest
import torch

from prime_rl.weight_transfer.adapter import make_hf_named_lazy_weights
from prime_rl.weight_transfer.chains import (
    UnsupportedOpError,
    region_elem_runs,
    resolve_chain_region,
    tensor_runs,
)
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight
from prime_rl.weight_transfer.sharding import route_region, zip_src_dst
from prime_rl.weight_transfer.wire import TrainerShard

E, I, H = 4, 6, 8


def shard_full_tensor(full: torch.Tensor, num_shards: int) -> tuple[list[torch.Tensor], list[TrainerShard]]:
    """Split a full tensor into ``num_shards`` contiguous dim-0 shard buffers,
    each a standalone contiguous tensor (a separate 'rank')."""
    rows = full.shape[0]
    per = (rows + num_shards - 1) // num_shards
    row_bytes = full[0].numel() * full.element_size()
    buffers, shards = [], []
    for agent in range(num_shards):
        start = agent * per
        count = min(per, rows - start)
        if count <= 0:
            break
        buf = full[start : start + count].contiguous()
        buffers.append(buf)  # keep alive
        shards.append(
            TrainerShard(
                agent=agent,
                row_start=start,
                num_rows=count,
                addr=buf.data_ptr(),
                row_bytes=row_bytes,
                device_id=0,
            )
        )
    return buffers, shards


def simulate_pull(trainer: dict[str, torch.Tensor], shards: dict[str, list[TrainerShard]], copies) -> int:
    """Move bytes the way the worker's NIXL READs would: resolve each chain to
    a region of the full logical tensor, route onto shards, zip against the dst
    runs, and memmove each unit."""
    moved = 0
    for copy in copies:
        full = trainer[copy.src_name]
        row_numel = full[0].numel()
        offset, shape, stride = resolve_chain_region(tuple(full.shape), full.dtype, copy.ops)
        src_runs = region_elem_runs(offset, shape, stride)
        src_pieces = route_region(src_runs, shards[copy.src_name], row_numel, full.element_size())
        for _agent, src_addr, dst_addr, nbytes in zip_src_dst(src_pieces, tensor_runs(copy.dst)):
            ctypes.memmove(dst_addr, (ctypes.c_char * nbytes).from_address(src_addr), nbytes)
            moved += nbytes
    return moved


@pytest.fixture
def trainer():
    torch.manual_seed(0)
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, H, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(8, H, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w1": torch.randn(E, I, H, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w2": torch.randn(E, H, I, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w3": torch.randn(E, I, H, dtype=torch.bfloat16),
        "model.layers.0.mlp.router.gate.weight": torch.randn(E, H, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(32, H, dtype=torch.bfloat16),
    }


def test_sharded_bake_and_pull_round_trip(trainer):
    # Trainer shards: experts split across 2 EP ranks, dense across 2 FSDP ranks.
    keep, shards = [], {}
    for name, full in trainer.items():
        bufs, sh = shard_full_tensor(full, 2)
        keep.extend(bufs)
        shards[name] = sh

    metas = [(name, t.dtype, tuple(t.shape)) for name, t in trainer.items()]
    recorder = BakeRecorder()
    lazy = dict(make_hf_named_lazy_weights(metas, torch.device("cpu"), recorder))

    qkv = torch.zeros(24, H, dtype=torch.bfloat16)
    w13 = torch.zeros(2, 2 * I, H, dtype=torch.bfloat16)
    w2 = torch.zeros(2, H, I, dtype=torch.bfloat16)
    gate = torch.zeros(E, H, dtype=torch.bfloat16)
    embed = torch.zeros(34, H, dtype=torch.bfloat16)
    local_experts = {1: 0, 3: 1}  # this worker owns global experts 1 and 3

    for name, weight in lazy.items():
        if name.endswith("q_proj.weight"):
            qkv.narrow(0, 0, 16).copy_(weight)
        elif name.endswith("k_proj.weight"):
            qkv.narrow(0, 16, 8).copy_(weight)
        elif ".experts." in name:
            expert_id = int(name.split(".experts.")[1].split(".")[0])
            if expert_id not in local_experts:
                continue
            local = local_experts[expert_id]
            if name.endswith("gate_proj.weight"):
                w13[local].narrow(0, 0, I).copy_(weight)
            elif name.endswith("up_proj.weight"):
                w13[local].narrow(0, I, I).copy_(weight)
            else:
                w2[local].copy_(weight)
        elif name.endswith("mlp.gate.weight"):
            gate.copy_(weight)
        elif name.endswith("embed_tokens.weight"):
            embed.narrow(0, 0, 32).copy_(weight)

    assert all(copy.dst.abs().sum() == 0 for copy in recorder.copies)  # bake moved no data
    simulate_pull(trainer, shards, recorder.copies)

    assert torch.equal(qkv[:16], trainer["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv[16:24], trainer["model.layers.0.self_attn.k_proj.weight"])
    for global_id, local in local_experts.items():
        assert torch.equal(w13[local, :I], trainer["model.layers.0.mlp.experts.w1"][global_id])
        assert torch.equal(w13[local, I:], trainer["model.layers.0.mlp.experts.w3"][global_id])
        assert torch.equal(w2[local], trainer["model.layers.0.mlp.experts.w2"][global_id])
    assert torch.equal(gate, trainer["model.layers.0.mlp.router.gate.weight"])
    assert torch.equal(embed[:32], trainer["model.embed_tokens.weight"])


def test_tp_source_narrow_across_shards(trainer):
    """A trailing-dim narrow (TP) on a dim-0-sharded source: each needed row
    still resolves to exactly one shard, narrowed on the trailing dim."""
    full = trainer["model.embed_tokens.weight"]
    _keep, shards = shard_full_tensor(full, 4)
    metas = [("model.embed_tokens.weight", torch.bfloat16, (32, H))]
    recorder = BakeRecorder()
    ((_, weight),) = make_hf_named_lazy_weights(metas, torch.device("cpu"), recorder)

    dst = torch.zeros(32, H // 2, dtype=torch.bfloat16)
    dst.copy_(weight.narrow(1, H // 2, H // 2))
    simulate_pull(trainer, {"model.embed_tokens.weight": shards}, recorder.copies)
    assert torch.equal(dst, full[:, H // 2 :])


def test_route_region_splits_across_shards():
    # full tensor (4, 2): rows 0-1 on agent 0, rows 2-3 on agent 1.
    shards = [
        TrainerShard(agent=0, row_start=0, num_rows=2, addr=1000, row_bytes=4, device_id=0),
        TrainerShard(agent=1, row_start=2, num_rows=2, addr=2000, row_bytes=4, device_id=0),
    ]
    # whole-tensor contiguous run of 8 elements (itemsize 2, row_numel 2)
    pieces = route_region([(0, 8)], shards, row_numel=2, itemsize=2)
    assert pieces == [(0, 1000, 8), (1, 2000, 8)]


def test_zip_src_dst_splits_at_boundaries():
    units = zip_src_dst([(0, 1000, 6), (1, 2000, 2)], [(500, 2), (600, 4), (700, 2)])
    assert units == [(0, 1000, 500, 2), (0, 1002, 600, 4), (1, 2000, 700, 2)]


def test_zip_src_dst_length_mismatch_raises():
    with pytest.raises(ValueError):
        zip_src_dst([(0, 0, 4)], [(0, 6)])


def test_unsupported_op_raises():
    recorder = BakeRecorder()
    weight = LazyWeight("w", torch.Size((4, 4)), torch.bfloat16, torch.device("cpu"), recorder)
    with pytest.raises(UnsupportedOpError):
        weight.float()
    with pytest.raises(UnsupportedOpError):
        weight + 1


def test_materializing_chain_rejected_at_resolve():
    with pytest.raises(UnsupportedOpError):
        resolve_chain_region((4, 6), torch.bfloat16, (("t", (), {}), ("contiguous", (), {})))
