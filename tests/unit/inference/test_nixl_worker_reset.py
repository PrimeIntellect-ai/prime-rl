"""The vLLM NIXL worker's side of a trainer replacement."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker import nixl as nixl_worker
from prime_rl.inference.vllm.worker.nixl import (
    NIXLWeightUpdateWorker,
    WeightTransferGroup,
    WeightTransferPlan,
    poll_abort,
)
from prime_rl.transports.weights.nixl.agent import NixlPeer


class FakeAgent:
    def __init__(self, fail_release: bool = False) -> None:
        self.fail_release = fail_release
        self.released: list[str] = []
        self.removed: list[str] = []

    def release_read(self, read: str) -> None:
        if self.fail_release:
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        self.released.append(read)

    def remove_remote_agent(self, peer: NixlPeer) -> None:
        self.removed.append(peer.remote_agent_name)


def make_worker(agent: FakeAgent) -> NIXLWeightUpdateWorker:
    worker = object.__new__(NIXLWeightUpdateWorker)
    worker.nixl_agent = agent
    worker.model_express = SimpleNamespace(rank=0)
    worker.stale_trainer_instances = set()
    worker.weight_transfer_plan = WeightTransferPlan(
        receive_arenas={},
        receive_buffer_count=1,
        trainer_peers=[NixlPeer(local_agent=agent, remote_agent_name="trainer-failed")],  # type: ignore[arg-type]
        groups=[WeightTransferGroup(name="layer.0", layers=[], pulls=["read-0", "read-1"])],  # type: ignore[list-item]
    )
    return worker


def test_reset_drops_the_plan_with_a_replaced_trainer():
    agent = FakeAgent()
    worker = make_worker(agent)
    worker.reset_weight_transfer(["failed"])
    assert worker.weight_transfer_plan is None
    assert worker.stale_trainer_instances == {"failed"}
    assert agent.released == ["read-0", "read-1"]
    assert agent.removed == ["trainer-failed"]
    worker.reset_weight_transfer(["older"])
    assert worker.stale_trainer_instances == {"failed", "older"}


def test_reset_survives_a_failed_release():
    agent = FakeAgent(fail_release=True)
    worker = make_worker(agent)
    worker.reset_weight_transfer(["failed"])
    assert worker.weight_transfer_plan is None
    assert agent.removed == ["trainer-failed"]


def test_abort_path_is_polled_at_most_every_interval(tmp_path, monkeypatch):
    abort_path = tmp_path / ".aborted"
    assert not poll_abort(None)()
    requested = poll_abort(str(abort_path))
    assert not requested()
    abort_path.touch()
    assert not requested()  # checked again only after ABORT_POLL_INTERVAL_S
    monkeypatch.setattr(nixl_worker, "ABORT_POLL_INTERVAL_S", 0.0)
    requested = poll_abort(str(abort_path))
    assert requested()
    abort_path.unlink()
    assert requested()  # an abort stays requested


def test_a_failed_arena_registration_releases_the_others(monkeypatch):
    """NIXL must not keep registrations of arenas the allocator can hand out again."""
    registered, deregistered = [], []

    class Agent:
        def register_tensor(self, tensor: torch.Tensor) -> str:
            if registered:
                raise RuntimeError("NIXL_ERR_BACKEND")
            registered.append(tensor)
            return "registration-0"

        def deregister(self, registration: str) -> None:
            deregistered.append(registration)

    monkeypatch.setattr(nixl_worker, "use_cuda_malloc_pool", contextlib.nullcontext)
    worker = object.__new__(NIXLWeightUpdateWorker)
    worker.nixl_agent = Agent()
    worker.device = torch.device("cpu")
    worker.receive_arenas, worker.receive_arena_registrations = {}, []
    with pytest.raises(RuntimeError, match="NIXL_ERR_BACKEND"):
        worker.allocate_receive_arenas({torch.float32: 4, torch.bfloat16: 4}, 2)
    assert deregistered == ["registration-0"]
    assert worker.receive_arena_registrations == [] and worker.receive_arenas == {}
