from pathlib import Path

import httpx

from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.utils.monitor.prime import PrimeMonitor


class _RaisingPrimeConfig:
    def __init__(self):
        raise FileNotFoundError("missing ~/.prime/config.json")


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self) -> dict:
        return self._payload


def test_prime_monitor_disables_cleanly_when_prime_config_missing(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.setattr("prime_rl.utils.monitor.prime.PrimeConfig", _RaisingPrimeConfig)

    monitor = PrimeMonitor(PrimeMonitorConfig(), output_dir=Path("."))

    assert monitor.enabled is False


def test_prime_monitor_registers_with_env_api_key_when_prime_config_missing(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "test-api-key")
    monkeypatch.setattr("prime_rl.utils.monitor.prime.PrimeConfig", _RaisingPrimeConfig)
    monkeypatch.setattr(PrimeMonitor, "_init_async_client", lambda self: None)

    def _post(url: str, headers: dict, json: dict, timeout: int) -> _Response:
        assert url == "https://api.primeintellect.ai/api/v1/rft/external-runs"
        assert headers == {"Authorization": "Bearer test-api-key"}
        assert json["team_id"] == "team-123"
        return _Response(201, {"run": {"id": "run-123"}})

    monkeypatch.setattr(httpx, "post", _post)

    monitor = PrimeMonitor(
        PrimeMonitorConfig(team_id="team-123", run_name="test-run"),
        output_dir=Path("."),
    )

    assert monitor.enabled is True
    assert monitor.run_id == "run-123"
