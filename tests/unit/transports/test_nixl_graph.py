import pytest
import torch

from prime_rl.transports.weights.nixl.graph import (
    Destination,
    LazyWeight,
    WeightLoadRecorder,
    apply_chain,
    plan_tensor_replay,
)


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
    expected = torch.full_like(destination, -1)
    expected[1:5:2, 1:7:2] = source[::2, ::2].float()
    replay_recorded_copies(recorder, {"weight": source}, destination, expected)


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
    assert lazy._ops[0].name == "cat"
    assert len(lazy._ops[0].args[0]) == 1
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

    storage = torch.full(tuple(2 * size + 2 for size in expected.shape), -1, dtype=expected.dtype)
    destination_index = tuple(slice(1, 1 + 2 * size, 2) for size in expected.shape)
    destination = storage[destination_index]
    recorder.active_destination = Destination(object(), "weight", storage)
    destination[...] = lazy
    torch.testing.assert_close(storage, torch.full_like(storage, -1))
    expected_storage = torch.full_like(storage, -1)
    expected_storage[destination_index] = expected
    replay_recorded_copies(recorder, sources, storage, expected_storage)


def replay_recorded_copies(recorder, sources, destination, expected):
    for copy in recorder.copies:
        assert all(operation.name != "cat" for operation in copy.ops)
        source = sources[copy.source_name]
        plan = plan_tensor_replay(tuple(source.shape), source.dtype, copy.ops)
        received = source.as_strided(plan.source_shape, plan.source_stride, plan.source_offset).clone()
        destination.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(
            apply_chain(received, plan.replay_ops)
        )
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [0, 1, 2, -1])
@pytest.mark.parametrize("source_count", [1, 2, 3])
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
def test_composed_concatenation_replay(dimension, source_count, operation):
    recorder = WeightLoadRecorder()
    sources = {
        name: torch.arange(index * 24, (index + 1) * 24, dtype=torch.bfloat16 if index == 0 else torch.float32).reshape(
            2, 3, 4
        )
        for index, name in enumerate("abc"[:source_count])
    }
    inputs = [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()]
    lazy = operation(torch.cat(inputs, dimension))
    expected = operation(torch.cat(list(sources.values()), dimension))
    destination = torch.full(expected.shape, -1, dtype=expected.dtype)
    recorder.active_destination = Destination(object(), "weight", destination)

    destination.copy_(lazy)

    torch.testing.assert_close(destination, torch.full_like(destination, -1))
    replay_recorded_copies(recorder, sources, destination, expected)


@pytest.mark.parametrize("concat", [torch.cat, torch.concat, torch.concatenate])
@pytest.mark.parametrize("single_input_cat", [False, True])
def test_nested_concatenation_replay(concat, single_input_cat):
    recorder = WeightLoadRecorder()
    sources = {
        name: torch.arange(index * 12, (index + 1) * 12, dtype=torch.float32).reshape(3, 4)
        for index, name in enumerate("abc")
    }
    inputs = [LazyWeight(name, value.shape, value.dtype, value.device, recorder) for name, value in sources.items()]
    a, b, c = inputs
    lazy = concat([concat([a, b], 1).t(), c.t()], 0)
    assert [operation.name for operation in lazy._ops] == ["cat", "t", "cat"]
    assert lazy._source_name == "a"
    real_a, real_b, real_c = sources.values()
    expected = concat([concat([real_a, real_b], 1).t(), real_c.t()], 0)
    lazy = lazy.reshape(3, 12)[:, 1::2].t()
    expected = expected.reshape(3, 12)[:, 1::2].t()
    if single_input_cat:
        lazy = concat([lazy]).view(-1)
        expected = concat([expected]).view(-1)
    destination = torch.full_like(expected, -1)
    recorder.active_destination = Destination(object(), "weight", destination)
    destination.copy_(lazy)
    replay_recorded_copies(recorder, sources, destination, expected)


def test_concatenation_transpose_and_cast_stay_two_copies():
    recorder = WeightLoadRecorder()
    inputs = [
        LazyWeight(name, torch.Size((2, 3, 4)), torch.bfloat16, torch.device("cpu"), recorder) for name in ("a", "b")
    ]
    lazy = torch.cat(inputs, dim=1).transpose(1, 2).float().contiguous()
    destination = torch.empty(lazy.shape, dtype=lazy.dtype, device="meta")
    recorder.active_destination = Destination(object(), "weight", destination)
    destination.copy_(lazy)
    assert len(recorder.copies) == 2
    assert {copy.source_name for copy in recorder.copies} == {"a", "b"}


def test_concatenation_preserves_cast_order():
    recorder = WeightLoadRecorder()
    sources = {
        name: torch.linspace(index + 0.001, index + 0.999, 24).reshape(2, 3, 4) for index, name in enumerate(("a", "b"))
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
    replay_recorded_copies(recorder, sources, destination, expected)
