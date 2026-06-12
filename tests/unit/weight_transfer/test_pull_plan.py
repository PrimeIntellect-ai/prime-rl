"""CPU tests of the sharded NIXL pull plan: lazy bake (through stamped loaders,
into meta destinations) -> chain resolution -> shard routing -> run matching,
with the RDMA reads simulated via ctypes against per-rank shard buffers.

The loaders mimic the vLLM patterns the bake must survive: fused-QKV
destination narrows, FusedMoE per-expert routing with EP skips, TP source
narrows, padded-vocab partial destinations. The trainer is modelled as dim-0
shards split across two ranks, so routing must split each region across the
owning shards.
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
    """Split a full tensor into ``num_shards`` contiguous dim-0 shard buffers."""
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
        buffers.append(buf)
        shards.append(
            TrainerShard(
                agent=agent, row_start=start, num_rows=count, addr=buf.data_ptr(), row_bytes=row_bytes, device_id=0
            )
        )
    return buffers, shards


def simulate_pull(trainer: dict[str, torch.Tensor], shards: dict, copies, real_params: dict[str, torch.Tensor]) -> int:
    """Move bytes the way the worker would: resolve each chain to a region of
    the full logical tensor, route onto shards, reconstruct the destination view
    from the recorded (offset, shape, stride) on the real param, and memmove."""
    moved = 0
    for c in copies:
        full = trainer[c.src_name]
        row_numel = full[0].numel()
        offset, shape, stride = resolve_chain_region(tuple(full.shape), full.dtype, c.ops)
        src_pieces = route_region(
            region_elem_runs(offset, shape, stride), shards[c.src_name], row_numel, full.element_size()
        )
        dst = real_params[c.param_name].as_strided(c.shape, c.stride, c.offset)
        for _agent, src_addr, dst_addr, nbytes in zip_src_dst(src_pieces, tensor_runs(dst)):
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


def _bake_with_stamps(lazy, loaders):
    """Drive the lazy loaders against META destinations, stamping the recorder
    so each copy_ records its (param_name, offset/shape/stride) — mirroring the
    worker's recording stamps over the layerwise-reload meta params."""
    recorder = list(lazy.values())[0]._recorder
    for pname, fn in loaders:
        recorder.current = (object(), pname)  # synthetic module; only param_name matters here
        fn()
        recorder.current = None


def test_sharded_bake_and_pull_round_trip(trainer):
    keep, shards = [], {}
    for name, full in trainer.items():
        bufs, sh = shard_full_tensor(full, 2)
        keep.extend(bufs)
        shards[name] = sh

    metas = [(name, t.dtype, tuple(t.shape)) for name, t in trainer.items()]
    recorder = BakeRecorder()
    lazy = dict(make_hf_named_lazy_weights(metas, torch.device("cpu"), recorder))

    # Destination params on META (as during layerwise-reload bake).
    qkv = torch.empty(24, H, dtype=torch.bfloat16, device="meta")
    w13 = torch.empty(2, 2 * I, H, dtype=torch.bfloat16, device="meta")
    w2 = torch.empty(2, H, I, dtype=torch.bfloat16, device="meta")
    gate = torch.empty(E, H, dtype=torch.bfloat16, device="meta")
    embed = torch.empty(34, H, dtype=torch.bfloat16, device="meta")
    local_experts = {1: 0, 3: 1}

    loaders = []
    for name, weight in lazy.items():
        if name.endswith("q_proj.weight"):
            loaders.append(("qkv", lambda w=weight: qkv.narrow(0, 0, 16).copy_(w)))
        elif name.endswith("k_proj.weight"):
            loaders.append(("qkv", lambda w=weight: qkv.narrow(0, 16, 8).copy_(w)))
        elif ".experts." in name:
            expert_id = int(name.split(".experts.")[1].split(".")[0])
            if expert_id not in local_experts:
                continue
            le = local_experts[expert_id]
            if name.endswith("gate_proj.weight"):
                loaders.append(("w13", lambda w=weight, le=le: w13[le].narrow(0, 0, I).copy_(w)))
            elif name.endswith("up_proj.weight"):
                loaders.append(("w13", lambda w=weight, le=le: w13[le].narrow(0, I, I).copy_(w)))
            else:
                loaders.append(("w2", lambda w=weight, le=le: w2[le].copy_(w)))
        elif name.endswith("mlp.gate.weight"):
            loaders.append(("gate", lambda w=weight: gate.copy_(w)))
        elif name.endswith("embed_tokens.weight"):
            loaders.append(("embed", lambda w=weight: embed.narrow(0, 0, 32).copy_(w)))

    _bake_with_stamps(lazy, loaders)

    # Reconstruct destinations on real storage and pull.
    real = {
        "qkv": torch.zeros(24, H, dtype=torch.bfloat16),
        "w13": torch.zeros(2, 2 * I, H, dtype=torch.bfloat16),
        "w2": torch.zeros(2, H, I, dtype=torch.bfloat16),
        "gate": torch.zeros(E, H, dtype=torch.bfloat16),
        "embed": torch.zeros(34, H, dtype=torch.bfloat16),
    }
    simulate_pull(trainer, shards, recorder.copies, real)

    assert torch.equal(real["qkv"][:16], trainer["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(real["qkv"][16:24], trainer["model.layers.0.self_attn.k_proj.weight"])
    for global_id, le in local_experts.items():
        assert torch.equal(real["w13"][le, :I], trainer["model.layers.0.mlp.experts.w1"][global_id])
        assert torch.equal(real["w13"][le, I:], trainer["model.layers.0.mlp.experts.w3"][global_id])
        assert torch.equal(real["w2"][le], trainer["model.layers.0.mlp.experts.w2"][global_id])
    assert torch.equal(real["gate"], trainer["model.layers.0.mlp.router.gate.weight"])
    assert torch.equal(real["embed"][:32], trainer["model.embed_tokens.weight"])


def test_copy_only_recorded_when_stamped():
    """A copy_ with no recorder stamp is not attributed (left to fallback)."""
    recorder = BakeRecorder()
    w = LazyWeight("x", torch.Size((4, H)), torch.bfloat16, torch.device("cpu"), recorder)
    dst = torch.empty(4, H, dtype=torch.bfloat16, device="meta")
    dst.copy_(w)  # no stamp
    assert recorder.copies == []
    recorder.current = (object(), "weight")
    dst.copy_(w)
    assert len(recorder.copies) == 1 and recorder.copies[0].param_name == "weight"


def test_route_region_splits_across_shards():
    shards = [
        TrainerShard(agent=0, row_start=0, num_rows=2, addr=1000, row_bytes=4, device_id=0),
        TrainerShard(agent=1, row_start=2, num_rows=2, addr=2000, row_bytes=4, device_id=0),
    ]
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
    w = LazyWeight("w", torch.Size((4, 4)), torch.bfloat16, torch.device("cpu"), recorder)
    with pytest.raises(UnsupportedOpError):
        w.float()
    with pytest.raises(UnsupportedOpError):
        w + 1


def test_materializing_chain_rejected_at_resolve():
    with pytest.raises(UnsupportedOpError):
        resolve_chain_region((4, 6), torch.bfloat16, (("t", (), {}), ("contiguous", (), {})))
