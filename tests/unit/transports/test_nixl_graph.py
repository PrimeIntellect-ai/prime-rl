import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader

from prime_rl.transports.weights.nixl.graph import (
    Destination,
    LazyWeight,
    UnsupportedOpError,
    WeightLoadRecorder,
    apply_chain,
    make_hf_lazy_weights,
    plan_tensor_replay,
)
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerGroup, TrainerTensor, TrainerTensorTable


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


@pytest.mark.parametrize("upstream", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
def test_qwen35_lazy_hf_export(upstream, multimodal):
    if upstream:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    else:
        from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

    config = (
        Qwen3_5MoeConfig(text_config={"num_hidden_layers": 1})
        if multimodal
        else Qwen3_5MoeTextConfig(num_hidden_layers=1)
    )
    prefix = f"model{'.language_model' if multimodal else ''}.layers.0.mlp"
    shapes = {
        f"{prefix}.router.gate.weight": (2, 4),
        f"{prefix}.shared_expert.output_gate.weight": (1, 4),
        f"{prefix}.experts.gate_proj": (2, 3, 4),
        f"{prefix}.experts.up_proj": (2, 3, 4),
        f"{prefix}.experts.down_proj": (2, 4, 3),
    }
    table = TrainerTensorTable(
        [], 1, [TrainerGroup("layer0", [TrainerTensor(name, "bfloat16", shape, []) for name, shape in shapes.items()])]
    )
    recorder = WeightLoadRecorder()
    weights = dict(make_hf_lazy_weights(table, device=torch.device("cpu"), recorder=recorder, hf_config=config))

    assert set(weights) == {
        f"{prefix}.gate.weight",
        f"{prefix}.shared_expert_gate.weight",
        f"{prefix}.experts.gate_up_proj",
        f"{prefix}.experts.down_proj",
    }
    assert weights[f"{prefix}.gate.weight"]._source_name == f"{prefix}.router.gate.weight"
    assert weights[f"{prefix}.shared_expert_gate.weight"]._source_name == f"{prefix}.shared_expert.output_gate.weight"
    combined = weights[f"{prefix}.experts.gate_up_proj"]
    assert combined.shape == (2, 6, 4)
    # vLLM deconcatenates the fused checkpoint before loading individual experts.
    for projection, part in zip(("gate", "up"), combined.chunk(2, dim=1), strict=True):
        for expert, weight in enumerate(part.unbind()):
            destination = torch.empty(weight.shape, dtype=weight.dtype)
            recorder.active_destination = Destination(object(), "weight", destination)
            recorder.copies.clear()
            destination.copy_(weight)
            (copy,) = recorder.copies
            assert copy.source_name == f"{prefix}.experts.{projection}_proj"
            source = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
            torch.testing.assert_close(apply_chain(source, copy.ops), source[expert])


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("operation", ["copy", "narrow", "chunk", "split", "other_dim", "empty"])
def test_lazy_concatenation_replay(dim, operation):
    recorder = WeightLoadRecorder()
    sources = {
        "a": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        "b": torch.arange(24, 48, dtype=torch.bfloat16).reshape(2, 3, 4),
    }
    lazy = torch.cat(
        [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()], dim=dim
    )
    expected = torch.cat(list(sources.values()), dim=dim)
    if operation == "narrow":
        slice_length = expected.shape[dim] - 2
        lazy = lazy.narrow(dim, 1, slice_length)
        expected = expected.narrow(dim, 1, slice_length)
    elif operation == "chunk":
        lazy = lazy.chunk(3, dim=dim)[1]
        expected = expected.chunk(3, dim=dim)[1]
    elif operation == "split":
        widths = [1, expected.shape[dim] - 2, 1]
        lazy = lazy.split(widths, dim=dim)[1]
        expected = expected.split(widths, dim=dim)[1]
    elif operation == "other_dim":
        other_dimension = (dim + 1) % 3
        lazy = lazy.narrow(other_dimension, -1, 1)
        expected = expected.narrow(other_dimension, -1, 1)
    elif operation == "empty":
        lazy = lazy.narrow(dim, lazy.shape[dim], 0)
        expected = expected.narrow(dim, expected.shape[dim], 0)

    destination = torch.full(expected.shape, -1, dtype=expected.dtype)
    recorder.active_destination = Destination(object(), "weight", destination)
    destination[...] = lazy
    torch.testing.assert_close(destination, torch.full_like(destination, -1))
    for copy in recorder.copies:
        source = sources[copy.source_name]
        plan = plan_tensor_replay(tuple(source.shape), source.dtype, copy.ops)
        received = source.as_strided(plan.source_shape, plan.source_stride, plan.source_offset).clone()
        destination.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(
            apply_chain(received, plan.replay_ops)
        )
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


def replay_recorded_regions(recorder, sources, destination, expected):
    for copy in recorder.copies:
        source = sources[copy.source_name]
        plan = plan_tensor_replay(tuple(source.shape), source.dtype, copy.ops)
        received = source.as_strided(plan.source_shape, plan.source_stride, plan.source_offset).clone()
        destination.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(
            apply_chain(received, plan.replay_ops)
        )
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [0, 1, 2, -1])
@pytest.mark.parametrize(
    "operation",
    [
        lambda value: value.float().contiguous().transpose(0, 2),
        lambda value: value[:, 1:, ::2],
        lambda value: value[None, ..., -1],
        lambda value: value.transpose(1, 2).reshape(3, -1),
        lambda value: value.flatten().view(4, -1).t(),
        lambda value: value.unsqueeze(1).squeeze(1),
        lambda value: value.reshape(-1, 1).squeeze(),
        lambda value: value.flatten()[3].reshape(()),
        lambda value: value.float().chunk(2, dim=-1)[0],
        lambda value: value.contiguous().split(1, dim=0)[0],
        lambda value: value.float().narrow(1, 0, 1),
        lambda value: value.permute(2, 0, 1),
        lambda value: value.transpose(0, 2).flatten(),
        lambda value: value.unsqueeze(0).unbind(0)[0],
        lambda value: value.flatten()[:0].reshape(0, 1),
    ],
)
def test_composed_concatenation_replay(dimension, operation):
    recorder = WeightLoadRecorder()
    sources = {
        "a": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        "b": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4),
    }
    inputs = [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()]
    lazy = operation(torch.cat(inputs, dimension))
    expected = operation(torch.cat(list(sources.values()), dimension))
    destination = torch.full(expected.shape, -1, dtype=expected.dtype)
    recorder.active_destination = Destination(object(), "weight", destination)

    destination.copy_(lazy)

    torch.testing.assert_close(destination, torch.full_like(destination, -1))
    replay_recorded_regions(recorder, sources, destination, expected)


@pytest.mark.parametrize("concat", [torch.cat, torch.concat, torch.concatenate])
def test_nested_concatenation_replay(concat):
    recorder = WeightLoadRecorder()
    sources = {
        name: torch.arange(index * 12, (index + 1) * 12, dtype=torch.float32).reshape(3, 4)
        for index, name in enumerate("abc")
    }
    inputs = [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()]
    a, b, c = inputs
    lazy = concat([concat([a, b], 1).t(), c.t()], 0)
    real_a, real_b, real_c = sources.values()
    expected = concat([concat([real_a, real_b], 1).t(), real_c.t()], 0)
    lazy = lazy.reshape(3, 12)[:, 1::2].t()
    expected = expected.reshape(3, 12)[:, 1::2].t()
    destination = torch.full_like(expected, -1)
    recorder.active_destination = Destination(object(), "weight", destination)
    destination.copy_(lazy)
    replay_recorded_regions(recorder, sources, destination, expected)


def test_concatenation_transpose_and_cast_stay_two_copies():
    recorder = WeightLoadRecorder()
    inputs = [
        LazyWeight(name, torch.Size((256, 512, 2048)), torch.bfloat16, torch.device("cpu"), recorder)
        for name in ("gate", "up")
    ]
    lazy = torch.cat(inputs, dim=1).transpose(1, 2).float().contiguous()
    destination = torch.empty(lazy.shape, dtype=lazy.dtype, device="meta")
    recorder.active_destination = Destination(object(), "weight", destination)
    destination.copy_(lazy)
    assert len(recorder.copies) == 2
    assert {copy.source_name for copy in recorder.copies} == {"gate", "up"}


def test_concatenation_casts_preserve_rounding_and_existing_replay_behavior():
    recorder = WeightLoadRecorder()
    sources = {
        name: torch.linspace(index + 0.001, index + 0.999, 24).reshape(2, 3, 4)
        for index, name in enumerate(("gate", "up"))
    }
    inputs = [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()]

    def transform(values):
        return torch.cat([value.bfloat16().float() for value in values], dim=1).bfloat16().float().unbind()[1]

    lazy = transform(inputs)
    expected = transform(list(sources.values()))
    destination = torch.full_like(expected, -1)
    recorder.active_destination = Destination(object(), "weight", destination)
    destination.copy_(lazy)
    assert len(recorder.copies) == 2
    for copy in recorder.copies:
        source = sources[copy.source_name]
        plan = plan_tensor_replay(tuple(source.shape), source.dtype, copy.ops)
        assert plan.source_offset == 0
        assert plan.source_shape == tuple(source.shape)
    replay_recorded_regions(recorder, sources, destination, expected)
