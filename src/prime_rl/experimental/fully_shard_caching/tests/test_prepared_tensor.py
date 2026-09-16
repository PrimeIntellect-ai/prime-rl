import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from prime_rl.experimental.fully_shard_caching.prepared_tensor import (
    PREPARE_CALLS,
    ShardedPreparedTensor,
    UnshardedPreparedTensor,
    install_prepared_weights,
    prepared_or_none,
)

EXPERTS = 4
OUT_FEATURES = 6
IN_FEATURES = 8


def scale_prepare(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "doubled": (weight * 2).contiguous(),
        "row_absmax": weight.abs().amax(dim=-1).contiguous(),
    }


def aliasing_prepare(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    doubled = (weight * 2).contiguous()
    return {"doubled": doubled, "same": doubled.view(-1)}


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
    install_prepared_weights(module, {"gate_proj": scale_prepare, "down_proj": scale_prepare})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    return module


def test_install_wraps_parameters_preserving_metadata(module):
    original = module.gate_proj.detach().clone()
    install_prepared_weights(module, {"gate_proj": scale_prepare})

    assert isinstance(module.gate_proj.data, ShardedPreparedTensor)
    assert module.gate_proj.shape == original.shape
    assert module.gate_proj.dtype == original.dtype
    assert torch.equal(module.gate_proj.data._tensor, original)
    assert not isinstance(module.down_proj.data, ShardedPreparedTensor)


def test_install_rejects_missing_parameter(module):
    with pytest.raises(ValueError, match="no parameter 'up_proj'"):
        install_prepared_weights(module, {"up_proj": scale_prepare})


def test_install_rejects_a_non_parameter_attribute(module):
    module.register_buffer("scale", torch.ones(EXPERTS))
    with pytest.raises(ValueError, match="no parameter 'scale'"):
        install_prepared_weights(module, {"scale": scale_prepare})


def test_install_rejects_already_wrapped_parameter(module):
    install_prepared_weights(module, {"gate_proj": scale_prepare})
    with pytest.raises(ValueError, match="already wrapped"):
        install_prepared_weights(module, {"gate_proj": scale_prepare})


def test_sharded_flatten_unflatten_round_trip(module):
    wrapped = ShardedPreparedTensor(module.gate_proj.data, scale_prepare)
    names, metadata = wrapped.__tensor_flatten__()
    restored = ShardedPreparedTensor.__tensor_unflatten__(
        {name: getattr(wrapped, name) for name in names}, metadata, wrapped.size(), wrapped.stride()
    )

    assert restored.prepare_fn is scale_prepare
    assert torch.equal(restored._tensor, wrapped._tensor)


def test_unsharded_flatten_unflatten_round_trip(module):
    prepared = scale_prepare(module.gate_proj.data)
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
    install_prepared_weights(module, {"gate_proj": scale_prepare})
    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        prepared_or_none(module.gate_proj.data)


def test_op_reading_a_resharded_fsdp_parameter_raises(sharded_module):
    sharded_module.reshard()
    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        prepared_or_none(sharded_module.gate_proj.data)


def test_install_rejects_keeping_the_gather_buffer(module):
    with pytest.raises(NotImplementedError, match="release_all_gather_outputs=False"):
        install_prepared_weights(module, {"gate_proj": scale_prepare}, release_all_gather_outputs=False)


def test_unwrapped_weight_reports_no_preparation(module):
    assert prepared_or_none(module.gate_proj) is None


def test_prepare_returning_aliased_entries_raises(module, single_rank_process_group):
    install_prepared_weights(module, {"gate_proj": aliasing_prepare})
    fully_shard(module, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="aliasing the same storage"):
        module.unshard()


def test_reading_an_unsharded_tensor_as_data_raises(sharded_module):
    sharded_module.unshard()
    with pytest.raises(RuntimeError, match="storage-free UnshardedPreparedTensor"):
        sharded_module.gate_proj.data.sum()


def test_distributing_a_wrapped_parameter_keeps_the_wrapper_inside(module, ep_mesh):
    install_prepared_weights(module, {"gate_proj": scale_prepare})
    original = module.gate_proj.data._tensor.clone()

    sharded = distribute_tensor(module.gate_proj, ep_mesh, [Shard(0)])

    assert isinstance(sharded, DTensor)
    assert isinstance(sharded._local_tensor, ShardedPreparedTensor)
    assert sharded._local_tensor.prepare_fn is scale_prepare
    assert torch.equal(sharded._local_tensor._tensor, original)


def test_prepared_or_none_looks_through_a_dtensor(module, ep_mesh):
    install_prepared_weights(module, {"gate_proj": scale_prepare})
    sharded = distribute_tensor(module.gate_proj, ep_mesh, [Shard(0)])

    with pytest.raises(RuntimeError, match="outside its weights' unshard scope"):
        prepared_or_none(sharded)

    local = sharded.to_local()._tensor
    prepared = scale_prepare(local)
    unsharded = DTensor.from_local(UnshardedPreparedTensor(local, prepared), ep_mesh, [Shard(0)], run_check=False)

    assert dict(prepared_or_none(unsharded)) == prepared
