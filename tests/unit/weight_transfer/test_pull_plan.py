"""End-to-end CPU tests of the NIXL pull plan: lazy bake -> chain
resolution -> run matching, with the RDMA byte moves simulated via ctypes.

Loaders below mimic the vLLM patterns the bake must survive: fused-QKV
destination narrows, FusedMoE per-expert routing with EP skips, source
narrows (TP shards), padded-vocab partial destinations.
"""

import ctypes

import pytest
import torch

from prime_rl.weight_transfer.adapter import make_hf_named_lazy_weights
from prime_rl.weight_transfer.chains import (
    UnsupportedOpError,
    contiguous_runs,
    match_runs,
    resolve_chain_region,
    tensor_runs,
)
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight

E, I, H = 4, 6, 8


def simulate_pull(trainer_tensors: dict[str, torch.Tensor], copies) -> int:
    """Move bytes from trainer tensors into the recorded destinations the way
    the worker's NIXL READs would: resolve each chain to a strided source
    region, decompose both sides into runs, and copy run by run."""
    moved = 0
    for copy in copies:
        src = trainer_tensors[copy.src_name]
        offset, shape, stride = resolve_chain_region(tuple(src.shape), src.dtype, copy.ops)
        src_runs = contiguous_runs(src.data_ptr(), src.element_size(), offset, shape, stride)
        dst_runs = tensor_runs(copy.dst)
        for src_addr, dst_addr, nbytes in match_runs(src_runs, dst_runs):
            ctypes.memmove(dst_addr, (ctypes.c_char * nbytes).from_address(src_addr), nbytes)
            moved += nbytes
    return moved


@pytest.fixture
def trainer_state_dict() -> dict[str, torch.Tensor]:
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


def test_bake_and_pull_round_trip(trainer_state_dict):
    metas = [(name, t.dtype, tuple(t.shape)) for name, t in trainer_state_dict.items()]
    recorder = BakeRecorder()
    lazy = dict(make_hf_named_lazy_weights(metas, torch.device("cpu"), recorder))

    # Stacked experts explode into per-expert HF names; the router is renamed.
    assert "model.layers.0.mlp.experts.2.up_proj.weight" in lazy
    assert "model.layers.0.mlp.gate.weight" in lazy
    assert "model.layers.0.mlp.router.gate.weight" not in lazy

    # vLLM-style destinations: fused qkv, EP-local fused w13/w2, padded vocab.
    qkv = torch.zeros(24, H, dtype=torch.bfloat16)
    w13 = torch.zeros(2, 2 * I, H, dtype=torch.bfloat16)
    w2 = torch.zeros(2, H, I, dtype=torch.bfloat16)
    gate = torch.zeros(E, H, dtype=torch.bfloat16)
    embed = torch.zeros(34, H, dtype=torch.bfloat16)
    local_experts = {1: 0, 3: 1}

    for name, weight in lazy.items():
        if name.endswith("q_proj.weight"):
            qkv.narrow(0, 0, 16).copy_(weight)
        elif name.endswith("k_proj.weight"):
            qkv.narrow(0, 16, 8).copy_(weight)
        elif ".experts." in name:
            expert_id = int(name.split(".experts.")[1].split(".")[0])
            if expert_id not in local_experts:
                continue  # EP: not local to this rank
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

    # The bake records but moves no data.
    assert all(copy.dst.abs().sum() == 0 for copy in recorder.copies)

    simulate_pull(trainer_state_dict, recorder.copies)

    sd = trainer_state_dict
    assert torch.equal(qkv[:16], sd["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv[16:24], sd["model.layers.0.self_attn.k_proj.weight"])
    for global_id, local in local_experts.items():
        assert torch.equal(w13[local, :I], sd["model.layers.0.mlp.experts.w1"][global_id])
        assert torch.equal(w13[local, I:], sd["model.layers.0.mlp.experts.w3"][global_id])
        assert torch.equal(w2[local], sd["model.layers.0.mlp.experts.w2"][global_id])
    assert torch.equal(gate, sd["model.layers.0.mlp.router.gate.weight"])
    assert torch.equal(embed[:32], sd["model.embed_tokens.weight"])


def test_source_narrow_chain(trainer_state_dict):
    """TP-style source slicing: loader narrows the placeholder before copy_."""
    metas = [("model.embed_tokens.weight", torch.bfloat16, (32, H))]
    recorder = BakeRecorder()
    ((_, weight),) = make_hf_named_lazy_weights(metas, torch.device("cpu"), recorder)

    shard = torch.zeros(32, H // 2, dtype=torch.bfloat16)
    shard.copy_(weight.narrow(1, H // 2, H // 2))

    simulate_pull(trainer_state_dict, recorder.copies)
    assert torch.equal(shard, trainer_state_dict["model.embed_tokens.weight"][:, H // 2 :])


def test_unsupported_op_raises():
    recorder = BakeRecorder()
    weight = LazyWeight("w", torch.Size((4, 4)), torch.bfloat16, torch.device("cpu"), recorder)
    with pytest.raises(UnsupportedOpError):
        weight.float()
    with pytest.raises(UnsupportedOpError):
        weight + 1


def test_copy_shape_mismatch_raises():
    recorder = BakeRecorder()
    weight = LazyWeight("w", torch.Size((4, 4)), torch.bfloat16, torch.device("cpu"), recorder)
    with pytest.raises(UnsupportedOpError):
        torch.zeros(2, 4, dtype=torch.bfloat16).copy_(weight)


def test_materializing_chain_rejected_at_resolve():
    with pytest.raises(UnsupportedOpError):
        resolve_chain_region((4, 6), torch.bfloat16, (("t", (), {}), ("contiguous", (), {})))


def test_match_runs_splits_at_boundaries():
    pairs = match_runs([(0, 6), (100, 2)], [(200, 2), (300, 4), (400, 2)])
    assert pairs == [(0, 200, 2), (2, 300, 4), (100, 400, 2)]
    assert sum(n for _, _, n in pairs) == 8


def test_match_runs_length_mismatch_raises():
    with pytest.raises(ValueError):
        match_runs([(0, 4)], [(0, 6)])
