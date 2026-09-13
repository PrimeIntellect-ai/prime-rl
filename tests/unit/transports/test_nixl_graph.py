import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader

from prime_rl.transports.weights.nixl.graph import (
    Destination,
    LazyWeight,
    UnsupportedOpError,
    WeightLoadRecorder,
    apply_chain,
    plan_tensor_replay,
)


@pytest.mark.parametrize("tp_rank", range(8))
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_lazy_qkv_sharded_assignment(tp_rank, device, dtype):
    channels = (2048, 2048, 4096)
    source = torch.arange(sum(channels) * 4, dtype=torch.float32).reshape(-1, 1, 4).to(dtype)
    destination = torch.full((sum(channels) // 8, 1, 4), -1, dtype=dtype, device=device)
    owner = Destination(object(), "weight", destination)
    recorder = WeightLoadRecorder(active_destination=owner)
    lazy = LazyWeight("conv1d.weight", source.shape, dtype, torch.device("cpu"), recorder)
    loader = mamba_v2_sharded_weight_loader([(n, 0, False) for n in channels], 8, tp_rank)

    loader(destination, lazy)

    assert len(recorder.copies) == 3
    if device == "cpu":
        torch.testing.assert_close(destination, torch.full_like(destination, -1))

    replayed = torch.full(destination.shape, -1, dtype=dtype)
    source_channel = destination_channel = 0
    for copy, width in zip(recorder.copies, channels, strict=True):
        local_width = width // 8
        plan = plan_tensor_replay(tuple(source.shape), dtype, copy.ops)
        assert copy.source_name == "conv1d.weight"
        assert copy.destination_module is owner.module
        assert copy.destination_name == "weight"
        assert copy.destination_offset == destination_channel * 4
        assert copy.destination_shape == (local_width, 1, 4)
        assert copy.destination_stride == destination.stride()
        assert copy.is_persistent == (device == "cpu")
        assert plan.source_offset == (source_channel + tp_rank * local_width) * 4
        assert plan.source_shape == (local_width, 1, 4)
        assert plan.replay_ops == ()

        received = source.as_strided(plan.source_shape, plan.source_stride, plan.source_offset).clone()
        replayed.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(
            apply_chain(received, plan.replay_ops)
        )
        source_channel += width
        destination_channel += local_width

    expected = torch.cat([block.chunk(8, dim=0)[tp_rank] for block in source.split(channels, dim=0)])
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("native", [False, True])
def test_lazy_copy_preserves_validation(native):
    recorder = WeightLoadRecorder()
    source = LazyWeight("weight", torch.Size((2, 4)), torch.float32, torch.device("cpu"), recorder)
    copy = torch.ops.aten.copy_.default if native else torch.Tensor.copy_

    with pytest.raises(UnsupportedOpError, match="copy_ between lazy graph tensors"):
        copy(source, source)
    with pytest.raises(UnsupportedOpError, match="copy_ shape mismatch"):
        copy(torch.empty(3, 4), source)
    with pytest.raises(UnsupportedOpError, match="only support BF16/FP32"):
        copy(torch.empty(2, 4, dtype=torch.float16), source)
    assert recorder.copies == []


@pytest.mark.parametrize("entrypoint", ["method", "native", "assignment"])
def test_lazy_copy_strided_destination(entrypoint):
    source = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    destination = torch.full((6, 8), -1.0)
    owner = Destination(object(), "weight", destination)
    recorder = WeightLoadRecorder()
    recorder.register_destination_storage(owner)
    lazy = LazyWeight("weight", source.shape, source.dtype, source.device, recorder)[::2, ::2].float()
    view = destination[1:5:2, 1:7:2]

    if entrypoint == "method":
        assert view.copy_(lazy) is view
    elif entrypoint == "native":
        assert torch.ops.aten.copy_.default(self=view, src=lazy) is view
    else:
        destination[1:5:2, 1:7:2] = lazy

    torch.testing.assert_close(destination, torch.full_like(destination, -1))
    (copy,) = recorder.copies
    assert copy.destination_offset == 9
    assert copy.destination_stride == (16, 2)
    assert copy.destination_shape == (2, 3)
    assert copy.is_persistent
    plan = plan_tensor_replay(tuple(source.shape), source.dtype, copy.ops)
    received = source.as_strided(plan.source_shape, plan.source_stride, plan.source_offset).clone()
    destination.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(
        apply_chain(received, plan.replay_ops)
    )
    expected = torch.full_like(destination, -1)
    expected[1:5:2, 1:7:2] = source[::2, ::2].float()
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


