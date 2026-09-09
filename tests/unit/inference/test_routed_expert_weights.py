import base64
from types import SimpleNamespace

import numpy as np
import pytest
from verifiers.v1.routing import RoutingData, complete_routing_capture, decode_routing_payload

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture, serialize_routed_experts
from prime_rl.orchestrator.trajectories import _encode_routed_experts
from prime_rl.trainer.batch import prepare_sample
from prime_rl.transports.batch.routing import validate_routed_experts
from prime_rl.transports.batch.types import TrainingSample


def arrays():
    ids = (np.arange(3 * 2 * 2).reshape(3, 2, 2) % 8).astype(np.int32)
    weights = np.broadcast_to(np.array([0.375, 0.625], dtype=np.float32), ids.shape).copy()
    return ids, weights


def test_serializer_legacy_payload_unchanged():
    ids, _ = arrays()
    payload = serialize_routed_experts(ids)
    assert set(payload) == {"data", "shape", "start", "dtype"}
    assert payload["dtype"] == "uint8"
    assert base64.b64decode(payload["data"]) == ids.astype(np.uint8).tobytes()


def test_serializer_dtype_stable_for_model_expert_range():
    ids, weights = arrays()
    payload = serialize_routed_experts(ids, weights=weights, num_experts=512)
    assert payload["dtype"] == "uint16"
    assert payload["format_version"] == 1
    assert base64.b64decode(payload["weights"]["data"]) == weights.tobytes()


@pytest.mark.parametrize("problem", ["no_ids", "wrong_shape", "bf16_like", "nan", "negative", "id_bound"])
def test_serializer_rejects_unpaired_or_invalid_data(problem):
    ids, weights = arrays()
    if problem == "no_ids":
        ids = None
    if problem == "wrong_shape":
        weights = weights[:-1]
    if problem == "bf16_like":
        weights = weights.astype(np.float16)
    if problem == "nan":
        weights[0, 0, 0] = float("nan")
    if problem == "negative":
        weights[0, 0, 0] = -0.5
    if problem == "id_bound":
        ids[0, 0, 0] = 8
    with pytest.raises(ValueError):
        serialize_routed_experts(ids, weights=weights, num_experts=8)


def test_http_to_verifier_to_batch_preserves_exact_coefficients():
    ids, weights = arrays()
    payload = serialize_routed_experts(ids, weights=weights, num_experts=8)
    routing, start = decode_routing_payload(payload, require_weights=True, num_experts=8)
    assert start == 0 and isinstance(routing, RoutingData)
    assert routing.weights.tobytes() == weights.tobytes()
    routing = complete_routing_capture(routing, total_tokens=4, completion_tokens=2)
    packed = _encode_routed_experts(routing, 4)
    sample = TrainingSample(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        logprobs=[0, 0, -0.1, -0.2],
        temperatures=[1] * 4,
        advantages=[0, 0, 1, 1],
        env_name="test",
        routed_experts=packed,
    )
    micro = prepare_sample(sample, seq_len=4)
    final_ids, final_weights, valid = validate_routed_experts(micro.routed_experts)
    assert final_ids[:3].tobytes() == ids.astype(np.uint8).tobytes()
    assert final_weights[:3].tobytes() == weights.tobytes()
    assert valid.tolist() == [True, True, True, False]
    assert not final_weights[-1].any()
    with pytest.raises(ValueError, match="align exactly"):
        _encode_routed_experts(routing, 3)


@pytest.mark.asyncio
async def test_capture_preserves_pair_and_avoids_discarded_encoding():
    ids, weights = arrays()
    output = SimpleNamespace(index=0, routed_experts=ids, routed_expert_weights=weights, finished=lambda: True)

    async def stream():
        yield SimpleNamespace(outputs=[output])

    capture = RoutedExpertsCapture(stream(), require_weights=True, num_experts=8)
    emitted = [item async for item in capture]
    assert len(emitted) == 1
    assert output.routed_experts is None
    assert output.routed_expert_weights is None
    assert base64.b64decode(capture.routed_experts[0]["weights"]["data"]) == weights.tobytes()


@pytest.mark.asyncio
async def test_capture_fails_when_finished_output_has_no_weights():
    ids, _ = arrays()

    async def stream():
        yield SimpleNamespace(outputs=[SimpleNamespace(index=0, routed_experts=ids, finished=lambda: True)])

    with pytest.raises(ValueError, match="missing paired"):
        async for _ in RoutedExpertsCapture(stream(), require_weights=True):
            pass


@pytest.mark.parametrize("start", [-1, True, 0.5, "0"])
def test_serializer_rejects_invalid_start_metadata(start):
    ids, weights = arrays()
    with pytest.raises(ValueError, match="nonnegative integer"):
        serialize_routed_experts(ids, weights=weights, start=start)


@pytest.mark.parametrize("num_experts", [0, 65537, True, 2.5, "8"])
def test_serializer_rejects_invalid_model_expert_count(num_experts):
    ids, weights = arrays()
    with pytest.raises(ValueError, match="1..65536"):
        serialize_routed_experts(ids, weights=weights, num_experts=num_experts)


@pytest.mark.parametrize("ids", [np.zeros((2, 2)), np.zeros((2, 0, 2), dtype=np.int32), np.zeros((2, 2, 2))])
def test_serializer_checks_id_metadata_without_asserts(ids):
    with pytest.raises(ValueError, match="shape|integer dtype"):
        serialize_routed_experts(ids)


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan"), 1.1])
def test_serializer_rejects_invalid_full_v1_coefficients(value):
    ids, weights = arrays()
    weights[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite and in"):
        serialize_routed_experts(ids, weights=weights)
