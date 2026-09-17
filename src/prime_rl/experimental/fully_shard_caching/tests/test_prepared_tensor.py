import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from prime_rl.experimental.fully_shard_caching.fp8_cast import (
    grouped_per_block_cast_to_fp8,
    grouped_per_block_cast_to_fp8_both_layouts,
)
from prime_rl.experimental.fully_shard_caching.ops import blockwise_fp8_prepare, row_scaled_prepare
from prime_rl.experimental.fully_shard_caching.prepared_tensor import (
    PREPARE_CALLS,
    ShardedPreparedTensor,
    UnshardedPreparedTensor,
    check_sharding_is_preparable,
    install_prepared_weights,
    unsharded_prepared_or_none,
    run_prepare,
)
from prime_rl.trainer.models.kernels.fp8_utils import ue8m0_for_device

EXPERTS = 4
OUT_FEATURES = 6
IN_FEATURES = 8
BLOCK_ALIGNED_SHAPE = (2, 256, 384)
RAGGED_SHAPE = (2, 200, 300)
SENTINEL_VALUE = 1e30
SENTINEL_BYTE = 0xA5


class ScaleOp:
    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if out is None:
            return {
                "doubled": (weight * 2).contiguous(),
                "row_absmax": weight.abs().amax(dim=-1).contiguous(),
            }
        torch.mul(weight, 2, out=out["doubled"])
        torch.amax(weight.abs(), dim=-1, out=out["row_absmax"])
        return out


class AliasingOp:
    def prepare(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        doubled = (weight * 2).contiguous()
        return {"doubled": doubled, "same": doubled.view(-1)}


class OutIgnoringOp:
    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return SCALE_OP.prepare(weight)


class OutRebindingOp:
    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out["doubled"] = (weight * 2).contiguous()
        return out


class ShardHalvingOp:
    """Halves each shard on its own, reducing the gathered rows to an absmax afterwards."""

    shard_blocking = (1, 1, 1)

    def prepare(self, weight: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        prepared = self.complete_gathered(self.prepare_shard(weight))
        if out is None:
            return prepared
        for name, tensor in prepared.items():
            out[name].copy_(tensor)
        return out

    def prepare_shard(
        self, shard: torch.Tensor, *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if out is None:
            return {"halved": (shard / 2).contiguous()}
        torch.div(shard, 2, out=out["halved"])
        return out

    def complete_gathered(
        self, wire: dict[str, torch.Tensor], *, out: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        halved = wire["halved"]
        if out is None:
            return {"halved": halved, "row_absmax": halved.abs().amax(dim=-1).contiguous()}
        torch.amax(halved.abs(), dim=-1, out=out["row_absmax"])
        return out


SCALE_OP = ScaleOp()
ALIASING_OP = AliasingOp()


class Weights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Parameter(torch.randn(EXPERTS, OUT_FEATURES, IN_FEATURES))
        self.down_proj = nn.Parameter(torch.randn(EXPERTS, IN_FEATURES, OUT_FEATURES))


@pytest.fixture
def module() -> Weights:
    torch.manual_seed(0)
    return Weights().cuda()


@pytest.fixture
def ep_mesh(single_rank_process_group) -> DeviceMesh:
    return init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))


@pytest.fixture
def sharded_module(module, single_rank_process_group) -> Weights:
    install_prepared_weights(module, {"gate_proj": SCALE_OP, "down_proj": SCALE_OP})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    return module


def test_install_wraps_parameters_preserving_metadata(module):
    original = module.gate_proj.detach().clone()
    install_prepared_weights(module, {"gate_proj": SCALE_OP})

    assert isinstance(module.gate_proj.data, ShardedPreparedTensor)
    assert module.gate_proj.shape == original.shape
    assert module.gate_proj.dtype == original.dtype
    assert torch.equal(module.gate_proj.data._tensor, original)
    assert not isinstance(module.down_proj.data, ShardedPreparedTensor)


def test_install_rejects_missing_parameter(module):
    with pytest.raises(ValueError, match="no parameter 'up_proj'"):
        install_prepared_weights(module, {"up_proj": SCALE_OP})


def test_install_rejects_a_non_parameter_attribute(module):
    module.register_buffer("scale", torch.ones(EXPERTS))
    with pytest.raises(ValueError, match="no parameter 'scale'"):
        install_prepared_weights(module, {"scale": SCALE_OP})


def test_install_rejects_already_wrapped_parameter(module):
    install_prepared_weights(module, {"gate_proj": SCALE_OP})
    with pytest.raises(ValueError, match="already wrapped"):
        install_prepared_weights(module, {"gate_proj": SCALE_OP})


def test_install_rejects_an_op_satisfying_neither_protocol(module):
    with pytest.raises(TypeError, match="neither a PrepareOp"):
        install_prepared_weights(module, {"gate_proj": SCALE_OP.prepare})


def test_sharded_flatten_unflatten_round_trip(module):
    wrapped = ShardedPreparedTensor(module.gate_proj.data, SCALE_OP)
    names, metadata = wrapped.__tensor_flatten__()
    restored = ShardedPreparedTensor.__tensor_unflatten__(
        {name: getattr(wrapped, name) for name in names}, metadata, wrapped.size(), wrapped.stride()
    )

    assert restored.op is SCALE_OP
    assert torch.equal(restored._tensor, wrapped._tensor)


def test_unsharded_flatten_unflatten_round_trip(module):
    prepared = SCALE_OP.prepare(module.gate_proj.data)
    wrapped = UnshardedPreparedTensor(module.gate_proj.data, prepared)
    names, metadata = wrapped.__tensor_flatten__()
    restored = UnshardedPreparedTensor.__tensor_unflatten__(
        {name: getattr(wrapped, name) for name in names}, metadata, wrapped.size(), wrapped.stride()
    )

    assert restored.shape == wrapped.shape
    assert set(restored.prepared) == set(prepared)
    for name, tensor in prepared.items():
        assert restored.prepared[name] is tensor


def test_prepare_runs_once_per_unshard(sharded_module):
    sharded_module.unshard()
    assert PREPARE_CALLS.count == 2

    sharded_module.unshard()
    assert PREPARE_CALLS.count == 2

    sharded_module.reshard()
    sharded_module.unshard()
    assert PREPARE_CALLS.count == 4


def test_refill_reuses_the_same_tensor_objects(sharded_module):
    sharded_module.unshard()
    first = dict(sharded_module.gate_proj.prepared)
    sizes = {name: tensor.untyped_storage().size() for name, tensor in first.items()}
    versions = {name: tensor._version for name, tensor in first.items()}

    sharded_module.reshard()
    sharded_module.unshard()
    second = sharded_module.gate_proj.prepared

    for name, tensor in first.items():
        assert second[name] is tensor
        assert second[name].untyped_storage().size() == sizes[name]
        assert second[name]._version == versions[name]


def test_optimizer_update_is_visible_after_reshard_and_unshard(sharded_module):
    sharded_module.unshard()
    before = sharded_module.gate_proj.prepared["doubled"].clone()
    sharded_module.reshard()

    with torch.no_grad():
        sharded_module.gate_proj.add_(1.0)

    sharded_module.unshard()
    after = sharded_module.gate_proj.prepared["doubled"]
    assert torch.allclose(after.float(), before.float() + 2.0, atol=5e-2)


def test_reshard_frees_the_prepared_storage(sharded_module):
    sharded_module.unshard()
    prepared = dict(sharded_module.gate_proj.prepared)
    sharded_module.reshard()

    assert all(tensor.untyped_storage().size() == 0 for tensor in prepared.values())


def test_op_reading_a_sharded_tensor_raises(module):
    install_prepared_weights(module, {"gate_proj": SCALE_OP})
    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        unsharded_prepared_or_none(module.gate_proj.data)


def test_op_reading_a_resharded_fsdp_parameter_raises(sharded_module):
    sharded_module.reshard()
    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        unsharded_prepared_or_none(sharded_module.gate_proj.data)


def test_unwrapped_weight_reports_no_preparation(module):
    assert unsharded_prepared_or_none(module.gate_proj) is None


def test_prepare_returning_aliased_entries_raises(module, single_rank_process_group):
    install_prepared_weights(module, {"gate_proj": ALIASING_OP})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="aliasing the same storage"):
        module.unshard()


def test_reading_an_unsharded_tensor_as_data_raises(sharded_module):
    sharded_module.unshard()
    with pytest.raises(RuntimeError, match="storage-free UnshardedPreparedTensor"):
        sharded_module.gate_proj.data.sum()


def test_distributing_a_wrapped_parameter_keeps_the_wrapper_inside(module, ep_mesh):
    install_prepared_weights(module, {"gate_proj": SCALE_OP})
    original = module.gate_proj.data._tensor.clone()

    sharded = distribute_tensor(module.gate_proj, ep_mesh, [Shard(0)])

    assert isinstance(sharded, DTensor)
    assert isinstance(sharded._local_tensor, ShardedPreparedTensor)
    assert sharded._local_tensor.op is SCALE_OP
    assert torch.equal(sharded._local_tensor._tensor, original)


def test_unsharded_prepared_or_none_looks_through_a_dtensor(module, ep_mesh):
    install_prepared_weights(module, {"gate_proj": SCALE_OP})
    sharded = distribute_tensor(module.gate_proj, ep_mesh, [Shard(0)])

    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        unsharded_prepared_or_none(sharded)

    local = sharded.to_local()._tensor
    prepared = SCALE_OP.prepare(local)
    unsharded = DTensor.from_local(UnshardedPreparedTensor(local, prepared), ep_mesh, [Shard(0)], run_check=False)

    assert dict(unsharded_prepared_or_none(unsharded).prepared) == prepared


def test_sharded_preparation_round_trips_through_fully_shard(module, single_rank_process_group):
    op = ShardHalvingOp()
    expected = op.prepare(module.gate_proj.detach().to(torch.bfloat16))
    install_prepared_weights(module, {"gate_proj": op})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

    module.unshard()
    assert PREPARE_CALLS.count == 1
    assert module.gate_proj.shape == (EXPERTS, OUT_FEATURES, IN_FEATURES)
    assert module.gate_proj.dtype == torch.bfloat16
    prepared = dict(module.gate_proj.prepared)
    assert set(prepared) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(prepared[name], tensor)

    module.reshard()
    module.unshard()
    assert PREPARE_CALLS.count == 2
    for name, tensor in expected.items():
        assert module.gate_proj.prepared[name] is prepared[name]
        assert torch.equal(prepared[name], tensor)


def test_multi_tensor_wire_on_a_size_one_mesh_raises(module, single_rank_process_group):
    class TwoTensorWireOp(ShardHalvingOp):
        def prepare_shard(self, shard, *, out=None):
            halved = super().prepare_shard(shard, out=out)
            if out is None:
                halved["doubled"] = (shard * 2).contiguous()
            return halved

    install_prepared_weights(module, {"gate_proj": TwoTensorWireOp()})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="drops every later input"):
        module.unshard()


@pytest.mark.parametrize(
    ("blocking", "outer_size", "mesh_size", "message"),
    [
        ((1, 1, None), torch.Size((6, 6, 8)), 4, "sharded unevenly"),
        ((None, 1, None), torch.Size((8, 6, 8)), 4, "reduces across dimension 0"),
        ((128, 1, None), torch.Size((256, 6, 8)), 4, "cuts a quantization tile"),
    ],
)
def test_check_sharding_is_preparable_rejects_shards_it_cannot_prepare(blocking, outer_size, mesh_size, message):
    with pytest.raises(ValueError, match=message):
        check_sharding_is_preparable(blocking, outer_size, 0, mesh_size, "Weights.gate_proj")


@pytest.mark.parametrize("prepare", [blockwise_fp8_prepare, row_scaled_prepare])
def test_filling_out_matches_allocating_bitwise(prepare):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    weight = torch.randn(BLOCK_ALIGNED_SHAPE, device="cuda", dtype=torch.bfloat16)
    reference = prepare(weight)

    out = {name: torch.empty_like(tensor) for name, tensor in reference.items()}
    for tensor in out.values():
        if tensor.dtype == torch.float8_e4m3fn:
            tensor.view(torch.uint8).fill_(SENTINEL_BYTE)
        else:
            tensor.fill_(SENTINEL_VALUE)
    handed_out = dict(out)

    filled = prepare(weight, out=out)

    assert filled is out
    for name, expected in reference.items():
        assert out[name] is handed_out[name]
        assert torch.equal(out[name].view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("op", [OutIgnoringOp(), OutRebindingOp()])
def test_prepare_that_does_not_fill_out_raises(op, module):
    weight = module.gate_proj.data
    out = SCALE_OP.prepare(weight)
    with pytest.raises(RuntimeError, match="replaced the tensors it was given"):
        run_prepare(op.prepare, weight, out=out)


@pytest.mark.parametrize("shape", [BLOCK_ALIGNED_SHAPE, RAGGED_SHAPE])
@pytest.mark.parametrize("fill_out", [False, True])
def test_both_layouts_matches_two_casts_bitwise(shape, fill_out):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    use_ue8m0 = ue8m0_for_device(weight.device)
    expected_out, expected_sf = grouped_per_block_cast_to_fp8(weight, use_ue8m0)
    expected_out_t, expected_sf_t = grouped_per_block_cast_to_fp8(weight.transpose(1, 2), use_ue8m0)

    if fill_out:
        expected = (expected_out, expected_sf, expected_out_t, expected_sf_t)
        destinations = [torch.empty_like(tensor) for tensor in expected]
        for tensor in destinations:
            if tensor.dtype == torch.float8_e4m3fn:
                tensor.view(torch.uint8).fill_(SENTINEL_BYTE)
            else:
                tensor.fill_(SENTINEL_VALUE)
        out, sf, out_t, sf_t = grouped_per_block_cast_to_fp8_both_layouts(
            weight, use_ue8m0, out=destinations[0], sf=destinations[1], out_t=destinations[2], sf_t=destinations[3]
        )
        for returned, destination in zip((out, sf, out_t, sf_t), destinations):
            assert returned is destination
    else:
        out, sf, out_t, sf_t = grouped_per_block_cast_to_fp8_both_layouts(weight, use_ue8m0)

    assert torch.equal(out.view(torch.uint8), expected_out.view(torch.uint8))
    assert torch.equal(out_t.view(torch.uint8), expected_out_t.view(torch.uint8))
    assert torch.equal(sf, expected_sf)
    assert torch.equal(sf_t, expected_sf_t)
