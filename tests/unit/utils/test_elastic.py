import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from prime_rl.utils.elastic import (
    AdapterState,
    ElasticInferencePool,
    check_server_model,
    discover_ready_servers,
    discover_server_ips,
)

# discover_server_ips tests


def test_discover_server_ips_returns_sorted_ips():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.return_value = ("hostname", [], ["10.0.0.3", "10.0.0.1", "10.0.0.2"])
        result = discover_server_ips("test.hostname")
        assert result == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


def test_discover_server_ips_returns_empty_list_on_dns_failure():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.side_effect = socket.gaierror("DNS lookup failed")
        result = discover_server_ips("nonexistent.hostname")
        assert result == []


def test_discover_server_ips_returns_single_ip():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.return_value = ("hostname", [], ["10.0.0.1"])
        result = discover_server_ips("single.hostname")
        assert result == ["10.0.0.1"]


# check_server_model tests


def test_check_server_model_returns_true_when_model_found():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        mock_client.get.return_value = mock_response

        has_model, is_healthy = asyncio.run(check_server_model("http://10.0.0.1:8000", "model-a"))

        assert has_model is True
        assert is_healthy is True


def test_check_server_model_returns_false_when_model_not_found():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "other-model"}]}
        mock_client.get.return_value = mock_response

        has_model, is_healthy = asyncio.run(check_server_model("http://10.0.0.1:8000", "model-a"))

        assert has_model is False
        assert is_healthy is True


def test_check_server_model_returns_false_on_connection_error():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        has_model, is_healthy = asyncio.run(check_server_model("http://10.0.0.1:8000", "model-a"))

        assert has_model is False
        assert is_healthy is False


def test_check_server_model_returns_false_on_http_error():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_client.get.return_value = mock_response

        has_model, is_healthy = asyncio.run(check_server_model("http://10.0.0.1:8000", "model-a"))

        assert has_model is False
        assert is_healthy is False


# discover_ready_servers tests


def test_discover_ready_servers_returns_servers_with_model_when_any_have_it():
    with (
        patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
        patch("prime_rl.utils.elastic.check_server_model") as mock_check,
    ):
        mock_discover.return_value = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

        async def mock_check_impl(url, model_name):
            if "10.0.0.1" in url:
                return True, True  # has model, healthy
            elif "10.0.0.2" in url:
                return False, True  # no model, healthy
            else:
                return False, True  # no model, healthy

        mock_check.side_effect = mock_check_impl

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert result == ["http://10.0.0.1:8000/v1"]


def test_discover_ready_servers_returns_empty_when_none_have_model():
    with (
        patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
        patch("prime_rl.utils.elastic.check_server_model") as mock_check,
    ):
        mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]
        mock_check.return_value = (False, True)  # no model, but healthy

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert result == []  # No servers have the model, so return empty


def test_discover_ready_servers_returns_empty_when_no_dns_records():
    with patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover:
        mock_discover.return_value = []

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert result == []


def test_discover_ready_servers_only_returns_servers_with_model():
    with (
        patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
        patch("prime_rl.utils.elastic.check_server_model") as mock_check,
    ):
        mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]

        async def mock_check_impl(url, model_name):
            if "10.0.0.1" in url:
                return True, True  # has model, healthy
            else:
                return False, True  # no model, healthy

        mock_check.side_effect = mock_check_impl

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        # Only returns servers that have the model
        assert result == ["http://10.0.0.1:8000/v1"]


# AdapterState tests


def test_adapter_state_creation():
    adapter = AdapterState(name="my-lora", path=Path("/weights/step_100"), step=100)
    assert adapter.name == "my-lora"
    assert adapter.path == Path("/weights/step_100")
    assert adapter.step == 100


# ElasticInferencePool adapter matching tests


def test_adapter_matches_when_no_adapter_desired():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )
        # No adapter desired (base model inference)
        assert pool._adapter_matches_desired(None) is True
        assert pool._adapter_matches_desired(AdapterState("x", Path("/x"), 0)) is True


def test_adapter_matches_by_path():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )
        pool._desired.path = Path("/weights/step_100")
        pool._desired.step = 100

        loaded = AdapterState(name="lora", path=Path("/weights/step_100"), step=100)
        assert pool._adapter_matches_desired(loaded) is True

        loaded_wrong_path = AdapterState(name="lora", path=Path("/weights/step_50"), step=50)
        assert pool._adapter_matches_desired(loaded_wrong_path) is False


def test_adapter_matches_by_step_when_nonzero():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )
        pool._desired.path = Path("/weights/step_100")
        pool._desired.step = 100

        # Different path but same step
        loaded = AdapterState(name="lora", path=Path("/other/path"), step=100)
        assert pool._adapter_matches_desired(loaded) is True


def test_adapter_does_not_match_by_zero_step():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )
        pool._desired.path = Path("/weights/step_0")
        pool._desired.step = 0

        # Step 0 should not match by step alone (avoid false positives)
        loaded = AdapterState(name="lora", path=Path("/other/path"), step=0)
        assert pool._adapter_matches_desired(loaded) is False


def test_adapter_returns_false_when_no_adapter_loaded():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )
        pool._desired.path = Path("/weights/step_100")
        pool._desired.step = 100

        assert pool._adapter_matches_desired(None) is False


# _get_loaded_adapter tests


def test_get_loaded_adapter_finds_correct_adapter_when_multiple_loaded():
    """Test that _get_loaded_adapter returns the adapter matching desired name, not the first one."""
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )

        # Set the desired adapter name
        pool._desired.name = "rft-target-run"
        pool._desired.path = Path("/data/outputs/target_run/broadcasts/step_10")
        pool._desired.step = 10

        # Mock admin client with multiple adapters (like the real scenario)
        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # Simulates vLLM response with multiple adapters from different runs
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "base-model",
                    "parent": None,
                    "root": "base-model",
                },
                {
                    "id": "rft-other-run",  # Different run's adapter (comes first!)
                    "parent": "base-model",
                    "root": "/data/outputs/other_run/broadcasts/step_48",
                },
                {
                    "id": "rft-target-run",  # Our desired adapter
                    "parent": "base-model",
                    "root": "/data/outputs/target_run/broadcasts/step_10",
                },
            ]
        }
        mock_admin.get.return_value = mock_response
        pool._admin_clients["10.0.0.1"] = mock_admin

        result = asyncio.run(pool._get_loaded_adapter("10.0.0.1"))

        # Should return our target adapter, not the first one found
        assert result is not None
        assert result.name == "rft-target-run"
        assert result.path == Path("/data/outputs/target_run/broadcasts/step_10")
        assert result.step == 10


def test_get_loaded_adapter_returns_none_when_desired_adapter_not_found():
    """Test that _get_loaded_adapter returns None when desired adapter is not in the list."""
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )

        pool._desired.name = "rft-missing-run"
        pool._desired.path = Path("/data/outputs/missing_run/broadcasts/step_5")
        pool._desired.step = 5

        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "base-model", "parent": None, "root": "base-model"},
                {"id": "rft-other-run", "parent": "base-model", "root": "/data/outputs/other_run/step_10"},
            ]
        }
        mock_admin.get.return_value = mock_response
        pool._admin_clients["10.0.0.1"] = mock_admin

        result = asyncio.run(pool._get_loaded_adapter("10.0.0.1"))

        assert result is None


def test_get_loaded_adapter_parses_step_from_path():
    """Test that _get_loaded_adapter correctly parses step number from path."""
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )

        pool._desired.name = "my-lora"

        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "my-lora", "parent": "base", "root": "/weights/step_42"},
            ]
        }
        mock_admin.get.return_value = mock_response
        pool._admin_clients["10.0.0.1"] = mock_admin

        result = asyncio.run(pool._get_loaded_adapter("10.0.0.1"))

        assert result is not None
        assert result.step == 42


def test_get_loaded_adapter_handles_step_dash_format():
    """Test that _get_loaded_adapter parses step-N format (with dash)."""
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            model_name="base-model",
            port=8000,
        )

        pool._desired.name = "my-lora"

        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "my-lora", "parent": "base", "root": "/weights/step-99"},
            ]
        }
        mock_admin.get.return_value = mock_response
        pool._admin_clients["10.0.0.1"] = mock_admin

        result = asyncio.run(pool._get_loaded_adapter("10.0.0.1"))

        assert result is not None
        assert result.step == 99
