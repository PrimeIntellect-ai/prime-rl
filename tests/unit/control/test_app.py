from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller, validate_update
from prime_rl.utils.config import dump_resolved_config


@pytest.fixture
def run():
    """A stand-in orchestrator: the controller reads and replaces its config."""
    run = SimpleNamespace(config=OrchestratorConfig(batch_size=64), revision=0)
    run.controller = Controller(
        requires_step=True,
        read_config=lambda: (run.revision, run.config),
        check_config=lambda raw: validate_update(run.config, raw),
    )
    run.client = TestClient(create_app(run.controller))
    return run


def update_request(config: OrchestratorConfig, **values) -> dict:
    return {"type": "update_config", "config": dump_resolved_config(config) | values}


def test_pause_request(run):
    response = run.client.post("/v1/actions", json={"type": "pause", "step": 3})

    assert response.status_code == 202
    assert response.json()["state"] == "pending"
    assert response.json()["step"] == 3


def test_update_config_applied(run):
    response = run.client.post("/v1/actions", json=update_request(run.config, batch_size=128))
    assert response.status_code == 202
    action_id = response.json()["id"]

    def apply(config, changes):
        run.config, run.revision = config, run.revision + 1

    run.controller.apply_update(5, apply)

    assert run.client.get(f"/v1/actions/{action_id}").json()["state"] == "applied"
    snapshot = run.client.get("/v1/config").json()
    assert snapshot["revision"] == 1
    assert snapshot["config"]["batch_size"] == 128


@pytest.mark.parametrize(
    ("values", "frozen", "invalid"),
    [
        ({"seq_len": 4096}, ["seq_len"], False),
        ({"dispatch_per_minute": 60}, ["dispatch_per_minute"], False),
        ({"batch_size": 0}, [], True),
    ],
    ids=["frozen", "frozen_not_applied_live", "invalid"],
)
def test_update_config_rejected(run, values, frozen, invalid):
    response = run.client.post("/v1/actions", json=update_request(run.config, **values))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["frozen"] == frozen
    assert bool(detail["errors"]) == invalid


def test_update_config_conflicts_with_queued_update(run):
    run.client.post("/v1/actions", json=update_request(run.config, max_staleness=2))

    response = run.client.post("/v1/actions", json=update_request(run.config, batch_size=128))

    assert response.status_code == 409
