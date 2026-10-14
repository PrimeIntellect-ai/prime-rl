"""The env server stamps the launcher's run label, not the bare run name, on every
sandbox its workers open."""

from types import SimpleNamespace

from prime_rl.entrypoints import env_server
from prime_rl.utils.sandboxes import RUN_LABEL_VAR


def test_the_env_server_labels_sandboxes_with_the_run_label(monkeypatch):
    served: dict = {}
    monkeypatch.setattr(env_server, "serve_env", lambda **kwargs: served.update(kwargs))
    monkeypatch.setattr(env_server, "env_config_data", lambda env: {})
    monkeypatch.setattr(env_server, "pool_serve_kwargs", lambda pool: {})
    monkeypatch.setattr(env_server, "setup_worker", lambda *args: served.setdefault("setup", args))
    monkeypatch.setenv("PRL_RUN_NAME", "my-run")
    monkeypatch.setenv(RUN_LABEL_VAR, "prl-my-run-0123456789")
    config = SimpleNamespace(
        address_file=None,
        serve=SimpleNamespace(pool=None, address="tcp://127.0.0.1:0", max_concurrent=1),
        log=SimpleNamespace(level="info", json_logging=False),
        env=None,
    )

    env_server.run_server.__wrapped__(config)
    served["log_setup"]()

    assert served["setup"][-1] == ["prl-my-run-0123456789"]
