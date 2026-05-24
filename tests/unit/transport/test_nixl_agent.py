"""Show that a NIXL agent's metadata + tensor descriptor make it through
Model Express end-to-end: register on the trainer side, publish, fetch
back from the inference side, byte-compare.
"""

from __future__ import annotations

import uuid

import pytest
import torch
from modelexpress.client import MxClient

from prime_rl.transport.mx_rendezvous import MxRendezvous

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="nixl agent needs CUDA"),
]
pytest.importorskip("nixl_cu13")

from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name  # noqa: E402


@pytest.fixture
def model_name() -> str:
    return f"prime-rl-test/{uuid.uuid4().hex}"


@pytest.fixture
def client(mx_server: str) -> MxClient:
    c = MxClient(server_url=mx_server)
    yield c
    c.close()


def test_publish_nixl_agent_metadata_through_mx(client, model_name):
    agent = NixlAgentWrapper(name=make_agent_name("trainer", 0))
    weight = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    agent.register_tensor(weight)

    metadata = agent.get_metadata()
    assert metadata, "expected non-empty NIXL agent metadata bytes"

    descriptor = agent.make_tensor_descriptor("model.layers.0.self_attn.qkv_proj.weight", weight)
    assert descriptor.addr == weight.data_ptr()
    assert descriptor.size == weight.numel() * weight.element_size()
    assert descriptor.dtype == "bfloat16"

    trainer = MxRendezvous(client=client, role="trainer", rank=0, peer_world_size=1, model_name=model_name)
    inference = MxRendezvous(client=client, role="inference", rank=0, peer_world_size=1, model_name=model_name)
    trainer.publish(nixl_metadata=metadata, tensors=[descriptor])
    inference.publish(nixl_metadata=b"inference-side-stub", tensors=[])

    [trainer_ref] = inference.wait_for_peers(timeout=5)
    fetched = inference.fetch_peer(trainer_ref)

    assert fetched.nixl_metadata == metadata
    assert len(fetched.tensors) == 1
    td = fetched.tensors[0]
    assert (td.name, td.addr, td.size, td.dtype) == (
        descriptor.name,
        descriptor.addr,
        descriptor.size,
        descriptor.dtype,
    )
