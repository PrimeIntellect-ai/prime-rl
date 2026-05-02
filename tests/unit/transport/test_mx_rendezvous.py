"""End-to-end tests for :class:`MxRendezvous` against a live Model Express server.

The ``mx_server`` fixture (in ``conftest.py``) brings up the stack via
docker-compose. Each test uses a unique model name so the Redis backend
(persistent across the session) doesn't cross-pollute peer counts.
"""

from __future__ import annotations

import uuid

import pytest
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

from prime_rl.transport.mx_rendezvous import MxRendezvous


@pytest.fixture
def model_name() -> str:
    """Unique synthetic model name per test so peer lists don't leak across tests."""
    return f"prime-rl-test/{uuid.uuid4().hex}"


@pytest.fixture
def client(mx_server: str) -> MxClient:
    c = MxClient(server_url=mx_server)
    yield c
    c.close()


def test_publish_returns_stable_mx_source_id(client, model_name):
    rdz = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    descriptor = p2p_pb2.TensorDescriptor(name="w0", addr=0, size=0, device_id=0, dtype="bfloat16")
    sid = rdz.publish(nixl_metadata=b"trainer-md", tensors=[descriptor])
    assert sid
    assert rdz.mx_source_id == sid
    # Re-publishing the same identity returns the same hash.
    assert rdz.publish(nixl_metadata=b"trainer-md", tensors=[descriptor]) == sid


def test_trainer_and_inference_have_distinct_mx_source_ids(client, model_name):
    """Role lives in extra_parameters, so the two roles hash to different ids."""
    trainer = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    inference = MxRendezvous(client=client, role="inference", rank=0, peer_world_size=1, model_name=model_name)
    t_sid = trainer.publish(nixl_metadata=b"t", tensors=[])
    i_sid = inference.publish(nixl_metadata=b"i", tensors=[])
    assert t_sid != i_sid


def test_cross_role_discovery(client, model_name):
    """2 trainers + 2 inference workers each find the other side and only the other side."""
    trainers = [
        MxRendezvous(client=client, role="trainer", rank=r, peer_world_size=2, model_name=model_name) for r in range(2)
    ]
    inferences = [
        MxRendezvous(client=client, role="inference", rank=r, peer_world_size=2, model_name=model_name)
        for r in range(2)
    ]
    for t in trainers:
        t.publish(nixl_metadata=f"t-{t.rank}".encode(), tensors=[])
    for i in inferences:
        i.publish(nixl_metadata=f"i-{i.rank}".encode(), tensors=[])

    t_peers = trainers[0].wait_for_peers(timeout=5)
    i_peers = inferences[0].wait_for_peers(timeout=5)

    assert {p.worker_rank for p in t_peers} == {0, 1}
    assert {p.worker_rank for p in i_peers} == {0, 1}
    assert {p.worker_id for p in t_peers} == {i.worker_id for i in inferences}
    assert {p.worker_id for p in i_peers} == {t.worker_id for t in trainers}


def test_fetch_peer_preserves_nixl_metadata(client, model_name):
    trainer = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    inference = MxRendezvous(client=client, role="inference", rank=0, peer_world_size=1, model_name=model_name)
    descriptor = p2p_pb2.TensorDescriptor(name="w", addr=0, size=0, device_id=0, dtype="bfloat16")
    trainer.publish(nixl_metadata=b"agent-bytes-from-trainer", tensors=[descriptor])
    inference.publish(nixl_metadata=b"agent-bytes-from-inference", tensors=[])

    [t_ref] = inference.wait_for_peers(timeout=5)
    t_meta = inference.fetch_peer(t_ref)
    assert t_meta.nixl_metadata == b"agent-bytes-from-trainer"
    assert {td.name for td in t_meta.tensors} == {"w"}


def test_wait_for_peers_times_out_when_none_arrive(client, model_name):
    rdz = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    rdz.publish(nixl_metadata=b"x", tensors=[])
    with pytest.raises(TimeoutError, match="inference"):
        rdz.wait_for_peers(timeout=1.5, poll_interval=0.5)


def test_status_filter_gates_discovery(client, model_name):
    """Inference only counts trainers in READY status."""
    trainer = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    inference = MxRendezvous(client=client, role="inference", rank=0, peer_world_size=1, model_name=model_name)
    trainer.publish(nixl_metadata=b"t", tensors=[])
    inference.publish(nixl_metadata=b"i", tensors=[])

    # Trainer hasn't called set_status(READY) yet — inference should time out
    # when it filters on READY.
    with pytest.raises(TimeoutError):
        inference.wait_for_peers(status=p2p_pb2.SOURCE_STATUS_READY, timeout=1.0, poll_interval=0.3)

    trainer.set_status(p2p_pb2.SOURCE_STATUS_READY)
    peers = inference.wait_for_peers(status=p2p_pb2.SOURCE_STATUS_READY, timeout=5)
    assert len(peers) == 1


def test_set_status_before_publish_raises(client, model_name):
    rdz = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    with pytest.raises(RuntimeError, match="publish"):
        rdz.set_status(p2p_pb2.SOURCE_STATUS_READY)
