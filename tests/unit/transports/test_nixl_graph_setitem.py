"""Tracer coverage for loaders that assign into a parameter slice instead of calling ``copy_``.

vLLM's ``mamba_v2_sharded_weight_loader`` loads ``linear_attn.conv1d`` that way, which used to
raise ``UnsupportedOpError`` and break NIXL transfer on every hybrid-attention model.
"""

import pytest
import torch

from prime_rl.transports.weights.nixl.graph import (
    Destination,
    LazyWeight,
    UnsupportedOpError,
    WeightLoadRecorder,
    apply_chain,
)

KERNEL = 4


@pytest.fixture
def recorder():
    return WeightLoadRecorder()


def _lazy(recorder, param, global_shape):
    """Point the recorder at ``param`` and hand back the lazy source a loader would receive."""
    destination = Destination(module=object(), name="conv1d.weight", tensor=param)
    recorder.register_destination_storage(destination)
    recorder.active_destination = destination
    return LazyWeight(
        "conv1d.weight",
        torch.Size(global_shape),
        torch.float32,
        torch.device("cpu"),
        recorder,
    )


def _replay(recorder, source, like):
    """Replay as the receiver does (``replay_tensor_copy`` in inference/vllm/worker/nixl.py)."""
    out = torch.zeros_like(like)
    for copy in recorder.copies:
        value = apply_chain(source, copy.ops)
        out.as_strided(copy.destination_shape, copy.destination_stride, copy.destination_offset).copy_(value)
    return out


def test_setitem_into_slice_is_recorded(recorder):
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))

    param.data[2:5] = lazy[2:5]

    assert len(recorder.copies) == 1
    source = torch.randn(8, 1, KERNEL)
    expected = torch.zeros_like(param)
    expected[2:5] = source[2:5]
    assert torch.equal(_replay(recorder, source, param), expected)


def test_setitem_of_view_as_source_is_recorded(recorder):
    """vLLM 0.32's mamba loader reshapes the source with ``view_as`` before assigning."""
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (16, 1, KERNEL))

    target = param.data[0:4, ...]
    param.data[0:4, ...] = lazy[4:8].view_as(target)

    assert len(recorder.copies) == 1
    source = torch.randn(16, 1, KERNEL)
    expected = torch.zeros_like(param)
    expected[0:4, ...] = source[4:8].view_as(target)
    assert torch.equal(_replay(recorder, source, param), expected)


def test_setitem_into_a_destination_that_is_itself_a_view(recorder):
    """A loader may narrow the parameter first, so the destination is itself a view."""
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))

    shard = param.data.narrow(0, 2, 4)
    shard[1:3] = lazy[3:5]

    assert len(recorder.copies) == 1
    source = torch.randn(8, 1, KERNEL)
    expected = torch.zeros_like(param)
    expected.narrow(0, 2, 4)[1:3] = source[3:5]
    assert torch.equal(_replay(recorder, source, param), expected)


def test_setitem_into_non_dim0_slice_replays_with_stride(recorder):
    """A non-dim-0 destination slice must round-trip through the strided replay."""
    param = torch.zeros(4, 6)
    lazy = _lazy(recorder, param, (4, 6))

    param.data[:, 1:3] = lazy[:, 1:3]

    assert len(recorder.copies) == 1
    source = torch.randn(4, 6)
    expected = torch.zeros_like(param)
    expected[:, 1:3] = source[:, 1:3]
    assert torch.equal(_replay(recorder, source, param), expected)


def test_setitem_shape_mismatch_is_reported_as_setitem(recorder):
    """Eager PyTorch would broadcast here; the recorded copy cannot."""
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))

    with pytest.raises(UnsupportedOpError, match="__setitem__ shape mismatch"):
        param.data[0:2] = lazy[0:4]


def test_setitem_must_target_a_view_of_the_destination(recorder):
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))

    # Advanced indexing copies, so the write cannot be expressed as a destination offset.
    with pytest.raises(UnsupportedOpError):
        param.data[torch.tensor([0, 2, 4])] = lazy[0:3]


def test_setitem_between_lazy_tensors_is_rejected(recorder):
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))
    other = _lazy(WeightLoadRecorder(), torch.zeros(8, 1, KERNEL), (8, 1, KERNEL))

    with pytest.raises(UnsupportedOpError):
        other[0:4] = lazy[0:4]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_matches_vllm_mamba_conv1d_loader(tp_size):
    """Replaying the trace must be byte-identical to vLLM's real loader (Qwen3.5 conv1d spec)."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader

    key_dim, value_dim = 16, 32
    spec = [(key_dim, 0, False), (key_dim, 0, False), (value_dim, 0, False)]
    full = key_dim * 2 + value_dim
    rows = full // tp_size

    for rank in range(tp_size):
        recorder = WeightLoadRecorder()
        param = torch.zeros(rows, 1, KERNEL)
        lazy = _lazy(recorder, param, (full, 1, KERNEL))

        mamba_v2_sharded_weight_loader(spec, tp_size, rank)(param, lazy)
        assert recorder.copies, f"tp={tp_size} rank={rank} recorded no copies"

        source = torch.randn(full, 1, KERNEL)
        expected = torch.zeros(rows, 1, KERNEL)
        mamba_v2_sharded_weight_loader(spec, tp_size, rank)(expected, source)

        assert torch.equal(_replay(recorder, source, param), expected), (
            f"replayed trace differs from the real loader at tp={tp_size} rank={rank}"
        )
