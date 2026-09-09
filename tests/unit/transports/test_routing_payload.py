import copy

import msgspec
import numpy as np
import pytest

from prime_rl.trainer.batch import build_bin_cost, prepare_batch, prepare_sample
from prime_rl.transports.batch.routing import (
    concatenate_routed_experts,
    copy_routed_experts,
    pad_routed_experts,
    routing_compatible,
    slice_routed_experts,
    validate_routed_experts,
)
from prime_rl.transports.batch.types import RoutedExperts, TrainingSample


def paired(n=4, dtype=np.uint8, terminal=True):
    ids = (np.arange(n * 2 * 2).reshape(n, 2, 2) % 8).astype(dtype)
    weights = np.broadcast_to(np.array([0.375, 0.625], dtype="<f4"), ids.shape).copy()
    valid = np.ones(n, dtype=np.bool_)
    if terminal and n:
        valid[-1] = False
        weights[-1] = 0
    return RoutedExperts(
        data=ids.tobytes(),
        shape=list(ids.shape),
        dtype=str(ids.dtype),
        weights=weights.tobytes(),
        valid=valid.tobytes(),
        format_version=1,
    )


def sample(routing):
    n = routing.shape[0]
    return TrainingSample(
        token_ids=list(range(10, 10 + n)),
        mask=[False] + [True] * (n - 1),
        logprobs=[0.0] * n,
        temperatures=[1.0] * n,
        advantages=[0.0] + [1.0] * (n - 1),
        env_name="test",
        routed_experts=routing,
    )


def test_legacy_positional_wire_is_compatible():
    raw = msgspec.msgpack.encode([b"\x01\x02", [2, 1, 1], "uint8"])
    decoded = msgspec.msgpack.decode(raw, type=RoutedExperts)
    assert decoded.weights is None and decoded.valid is None and decoded.format_version == 0
    ids, weights, valid = validate_routed_experts(decoded)
    assert ids.flatten().tolist() == [1, 2]
    assert weights is valid is None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_pair_exact_wire_roundtrip(dtype):
    record = paired(dtype=dtype)
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(record), type=RoutedExperts)
    assert restored == record
    ids, weights, valid = validate_routed_experts(restored)
    assert weights.tobytes() == record.weights
    assert not ids.flags.writeable and not weights.flags.writeable and not valid.flags.writeable
    assert valid.tolist() == [True, True, True, False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [4, 2]),
        ("shape", [4, 0, 2]),
        ("shape", [True, 2, 2]),
        ("data", b""),
        ("dtype", "object"),
        ("weights", b""),
        ("valid", None),
        ("valid", b"\x01\x02\x01\x00"),
        ("format_version", 2),
    ],
)
def test_malformed_pair_rejected(field, value):
    record = paired()
    setattr(record, field, value)
    with pytest.raises(ValueError):
        validate_routed_experts(record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_coefficients_rejected(value):
    record = paired()
    weights = np.frombuffer(record.weights, dtype="<f4").copy()
    weights[0] = value
    record.weights = weights.tobytes()
    with pytest.raises(ValueError, match="finite"):
        validate_routed_experts(record)


def test_invalid_row_cannot_have_nonzero_coefficient():
    record = paired()
    record.valid = bytes(4)
    with pytest.raises(ValueError, match="placeholder"):
        validate_routed_experts(record)


def test_pair_copy_slice_concat_keep_axes_and_storage():
    source = paired(6)
    copied = copy_routed_experts(source)
    assert copied.data is source.data and copied.weights is source.weights
    assert copied.shape is not source.shape
    a = slice_routed_experts(source, 0, 3)
    b = slice_routed_experts(source, 3, 6)
    assert concatenate_routed_experts([a, b]) == source
    ids, weights, valid = validate_routed_experts(b)
    original_ids, original_weights, original_valid = validate_routed_experts(source)
    assert ids.tobytes() == original_ids[3:].tobytes()
    assert weights.tobytes() == original_weights[3:].tobytes()
    assert valid.tobytes() == original_valid[3:].tobytes()


def test_mixed_modes_and_layouts_are_not_packed_together():
    full = paired()
    legacy = RoutedExperts(full.data, list(full.shape), full.dtype)
    assert not routing_compatible(full, legacy)
    with pytest.raises(ValueError, match="different"):
        concatenate_routed_experts([full, legacy])
    groups = prepare_batch(
        [sample(full), sample(legacy)], seq_len=8, num_train_workers=1, bin_cost=build_bin_cost(None)
    )
    assert len(groups[0]) == 2


def test_full_padding_does_not_modify_source_or_concentrate_every_slot_on_zero():
    source = paired()
    before = copy.deepcopy(source)
    padded = pad_routed_experts(source, 4)
    ids, weights, valid = validate_routed_experts(padded)
    assert source == before
    assert ids.shape == (8, 2, 2)
    assert np.all(weights[4:] == 0) and not valid[4:].any()
    assert (ids[4:] < 8).all() and np.unique(ids[4:]).size == 8
    assert np.all(ids[4:, :, 0] != ids[4:, :, 1])
    assert padded.weights[: len(source.weights)] == source.weights


def test_legacy_padding_is_unchanged():
    full = paired()
    legacy = RoutedExperts(full.data, list(full.shape), full.dtype)
    padded = pad_routed_experts(legacy, 2)
    ids, weights, valid = validate_routed_experts(padded)
    assert np.all(ids[4:] == 0)
    assert weights is valid is None


def test_crop_preserves_pair_and_rejects_missing_interior_before_crop():
    source = paired(6)
    cropped = prepare_sample(sample(source), seq_len=3)
    ids, weights, valid = validate_routed_experts(cropped.routed_experts)
    assert ids.shape == (3, 2, 2) and valid.all()
    assert weights.tobytes() == source.weights[: 3 * 2 * 2 * 4]
    source.valid = b"\x01\x00\x01\x01\x01\x00"
    weights = np.frombuffer(source.weights, dtype="<f4").copy().reshape(source.shape)
    weights[1] = 0
    source.weights = weights.tobytes()
    with pytest.raises(ValueError, match="final unforwarded"):
        prepare_sample(sample(source), seq_len=1)


def test_pack_roundtrip_and_padding_keep_per_sequence_validity():
    a, b = paired(3), paired(4)
    batches = prepare_batch(
        [sample(a), sample(b)], seq_len=12, num_train_workers=1, bin_cost=build_bin_cost(None), pad_to_multiple_of=8
    )[0]
    assert len(batches) == 1
    batch = batches[0]
    ids, weights, valid = validate_routed_experts(batch.routed_experts)
    assert len(batch.input_ids) == len(valid) == 8
    assert sum(batch.sequence_lengths) == sum(batch.seq_lens) == 8
    assert not valid[-1]
    # Order is packing-dependent. Every source contribution remains an exact pair.
    offset = 0
    for length in batch.sequence_lengths:
        real_len = length - 1 if offset + length == 8 else length
        expected = a if real_len == 3 else b
        row = slice_routed_experts(batch.routed_experts, offset, offset + real_len)
        assert row == expected
        offset += length
    assert np.all(weights[-1] == 0)
