import os

import pytest

from prime_rl.utils.process import set_modelexpress_port_defaults


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("MX_METADATA_PORT", "MX_WORKER_GRPC_PORT"):
        monkeypatch.delenv(name, raising=False)


def listeners(monkeypatch, gpus: str, *, trainer: bool, devices: int) -> set[int]:
    """Ports one process binds: ModelExpress adds each CUDA device index, counted from 0, to its base ports."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus)
    set_modelexpress_port_defaults(trainer=trainer)
    bases = [int(os.environ.pop("MX_METADATA_PORT"))] + (
        [] if trainer else [int(os.environ.pop("MX_WORKER_GRPC_PORT"))]
    )
    return {base + device for base in bases for device in range(devices)}


def test_trainer_and_engines_sharing_a_node_bind_disjoint_ports(monkeypatch):
    # A single-node run: an engine with TP2 on GPUs 0-1, another on 2-3, and the trainer on 4-7.
    first_engine = listeners(monkeypatch, "0,1", trainer=False, devices=2)
    second_engine = listeners(monkeypatch, "2,3", trainer=False, devices=2)
    trainer = listeners(monkeypatch, "4,5,6,7", trainer=True, devices=4)

    assert len(first_engine | second_engine | trainer) == 4 + 4 + 4
    assert 5555 not in first_engine | second_engine | trainer
