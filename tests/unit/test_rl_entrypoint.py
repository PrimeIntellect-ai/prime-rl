import tomllib
from types import SimpleNamespace

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import advertise_inference_to_external_envs, rl_local


def _config(
    *,
    address: str | None,
    base_url: list[str],
    admin_base_url: list[str] | None = None,
    managed_inference: bool = True,
):
    client = SimpleNamespace(base_url=base_url, admin_base_url=admin_base_url)
    env = SimpleNamespace(address=address)
    orchestrator = SimpleNamespace(
        model=SimpleNamespace(client=client),
        train=SimpleNamespace(env=[env]),
        eval=None,
    )
    return SimpleNamespace(
        inference=object() if managed_inference else None,
        orchestrator=orchestrator,
    )


def test_advertises_loopback_inference_to_external_envs():
    config = _config(
        address="tcp://env-node:5000",
        base_url=["http://localhost:8000/v1", "http://inference-b:8000/v1"],
    )

    changed = advertise_inference_to_external_envs(config, "inference-a")

    assert changed
    assert config.orchestrator.model.client.base_url == [
        "http://inference-a:8000/v1",
        "http://inference-b:8000/v1",
    ]
    assert config.orchestrator.model.client.admin_base_url == [
        "http://localhost:8000/v1",
        "http://inference-b:8000/v1",
    ]


def test_does_not_change_local_env_launches():
    config = _config(address=None, base_url=["http://localhost:8000/v1"])

    assert not advertise_inference_to_external_envs(config, "inference-a")
    assert config.orchestrator.model.client.base_url == ["http://localhost:8000/v1"]
    assert config.orchestrator.model.client.admin_base_url is None


def test_preserves_explicit_admin_urls():
    config = _config(
        address="tcp://env-node:5000",
        base_url=["http://127.0.0.1:8000/v1"],
        admin_base_url=["http://admin:9000/v1"],
    )

    assert advertise_inference_to_external_envs(config, "inference-a")
    assert config.orchestrator.model.client.admin_base_url == ["http://admin:9000/v1"]


def test_does_not_change_external_inference_config():
    config = _config(
        address="tcp://env-node:5000",
        base_url=["http://localhost:8000/v1"],
        managed_inference=False,
    )

    assert not advertise_inference_to_external_envs(config, "inference-a")
    assert config.orchestrator.model.client.base_url == ["http://localhost:8000/v1"]


def test_local_launcher_serializes_allocated_host_for_external_envs(tmp_path, monkeypatch):
    config = RLConfig.model_validate(
        {
            "output_dir": tmp_path,
            "dry_run": True,
            "trainer": {},
            "orchestrator": {
                "train": {"env": [{"id": "dummy", "address": "tcp://env-node:5000"}]},
            },
            "inference": {},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 2,
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            },
        }
    )
    monkeypatch.setattr("prime_rl.entrypoints.rl.socket.gethostname", lambda: "inference-a")

    rl_local(config)

    with (tmp_path / "configs" / "orchestrator.toml").open("rb") as f:
        orchestrator = tomllib.load(f)
    client = orchestrator["model"]["client"]
    assert client["base_url"] == ["http://inference-a:8000/v1"]
    assert client["admin_base_url"] == ["http://localhost:8000/v1"]
