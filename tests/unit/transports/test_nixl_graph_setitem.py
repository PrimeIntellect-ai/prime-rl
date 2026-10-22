"""Tracer coverage for weight loaders that assign into a slice instead of calling ``copy_``.

``LazyWeight.__torch_function__`` special-cases ``copy_``, so a loader writing
``param.data[a:b] = loaded[c:d]`` used to fall through to ``__torch_dispatch__`` and raise
``UnsupportedOpError: aten.copy_.default``. Every hybrid-attention model hits this: vLLM's
``mamba_v2_sharded_weight_loader`` loads ``linear_attn.conv1d`` that way, so NIXL weight
transfer failed outright on Qwen3.5-class models.
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
    """Apply the recorded copies to ``source``, the way the receiver does."""
    out = torch.zeros_like(like)
    flat = out.reshape(-1)
    for copy in recorder.copies:
        value = apply_chain(source, copy.ops)
        flat[copy.destination_offset : copy.destination_offset + value.numel()] = value.reshape(-1)
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
    """vLLM 0.32's mamba loader reshapes the source with ``view_as`` before assigning.

    ``view_as`` takes a tensor argument, so it is normalized to ``view`` with a concrete
    shape rather than recording a live tensor into the replay chain.
    """
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (16, 1, KERNEL))

    target = param.data[0:4, ...]
    param.data[0:4, ...] = lazy[4:8].view_as(target)

    assert len(recorder.copies) == 1
    source = torch.randn(16, 1, KERNEL)
    expected = torch.zeros_like(param)
    expected[0:4, ...] = source[4:8].view_as(target)
    assert torch.equal(_replay(recorder, source, param), expected)


def test_setitem_must_target_a_view_of_the_destination(recorder):
    param = torch.zeros(8, 1, KERNEL)
    lazy = _lazy(recorder, param, (8, 1, KERNEL))

    # Advanced indexing copies rather than viewing, so the write cannot be expressed as an
    # offset into the destination -- it must fail loudly instead of recording a wrong offset.
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
    """Replaying the trace must be byte-identical to running vLLM's real loader.

    The shard spec is Qwen3.5's actual ``linear_attn.conv1d`` layout
    (key, key, value -- all dim 0, no duplication).
    """
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
