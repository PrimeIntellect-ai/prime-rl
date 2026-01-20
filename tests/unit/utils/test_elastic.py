import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from prime_rl.utils.elastic import (
    ElasticInferencePool,
    LoadedAdapter,
    WorkerServerDiscovery,
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


def test_discover_ready_servers_returns_all_healthy_when_none_have_model():
    with (
        patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
        patch("prime_rl.utils.elastic.check_server_model") as mock_check,
    ):
        mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]
        mock_check.return_value = (False, True)  # no model, but healthy

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert set(result) == {"http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"}


def test_discover_ready_servers_returns_empty_when_no_dns_records():
    with patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover:
        mock_discover.return_value = []

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert result == []


def test_discover_ready_servers_excludes_unhealthy_servers():
    with (
        patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
        patch("prime_rl.utils.elastic.check_server_model") as mock_check,
    ):
        mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]

        async def mock_check_impl(url, model_name):
            if "10.0.0.1" in url:
                return False, True  # healthy
            else:
                return False, False  # unhealthy

        mock_check.side_effect = mock_check_impl

        result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

        assert result == ["http://10.0.0.1:8000/v1"]


# LoadedAdapter tests


def test_loaded_adapter_creation():
    adapter = LoadedAdapter(name="my-lora", path=Path("/weights/step_100"), step=100)
    assert adapter.name == "my-lora"
    assert adapter.path == Path("/weights/step_100")
    assert adapter.step == 100


# ElasticInferencePool adapter matching tests


def test_adapter_matches_when_no_adapter_desired():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            port=8000,
        )
        # No adapter desired (base model inference)
        assert pool._adapter_matches_desired(None) is True
        assert pool._adapter_matches_desired(LoadedAdapter("x", Path("/x"), 0)) is True


def test_adapter_matches_by_path():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            port=8000,
        )
        pool._desired.lora_path = Path("/weights/step_100")
        pool._desired.step = 100

        loaded = LoadedAdapter(name="lora", path=Path("/weights/step_100"), step=100)
        assert pool._adapter_matches_desired(loaded) is True

        loaded_wrong_path = LoadedAdapter(name="lora", path=Path("/weights/step_50"), step=50)
        assert pool._adapter_matches_desired(loaded_wrong_path) is False


def test_adapter_matches_by_step_when_nonzero():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            port=8000,
        )
        pool._desired.lora_path = Path("/weights/step_100")
        pool._desired.step = 100

        # Different path but same step
        loaded = LoadedAdapter(name="lora", path=Path("/other/path"), step=100)
        assert pool._adapter_matches_desired(loaded) is True


def test_adapter_does_not_match_by_zero_step():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            port=8000,
        )
        pool._desired.lora_path = Path("/weights/step_0")
        pool._desired.step = 0

        # Step 0 should not match by step alone (avoid false positives)
        loaded = LoadedAdapter(name="lora", path=Path("/other/path"), step=0)
        assert pool._adapter_matches_desired(loaded) is False


def test_adapter_returns_false_when_no_adapter_loaded():
    with patch("prime_rl.utils.elastic.get_logger"):
        pool = ElasticInferencePool(
            hostname="test.hostname",
            client_config=MagicMock(),
            port=8000,
        )
        pool._desired.lora_path = Path("/weights/step_100")
        pool._desired.step = 100

        assert pool._adapter_matches_desired(None) is False


# WorkerServerDiscovery tests


def test_worker_server_discovery_initialization():
    with patch("prime_rl.utils.elastic.get_logger"):
        mock_config = MagicMock()
        mock_config.elastic.hostname = "test.hostname"
        mock_config.elastic.port = 8000
        mock_config.elastic.sync_interval = 5.0

        discovery = WorkerServerDiscovery(mock_config, "my-model")

        assert discovery._hostname == "test.hostname"
        assert discovery._port == 8000
        assert discovery._sync_interval == 5.0
        assert discovery._model_name == "my-model"
        assert discovery.clients == []


def test_worker_server_discovery_refresh_creates_clients_on_discovery():
    with (
        patch("prime_rl.utils.elastic.get_logger"),
        patch("prime_rl.utils.elastic.discover_ready_servers") as mock_discover,
    ):
        mock_config = MagicMock()
        mock_config.elastic.hostname = "test.hostname"
        mock_config.elastic.port = 8000
        mock_config.elastic.sync_interval = 0.0  # No throttling for tests
        mock_config.timeout = 30
        mock_config.api_key_var = "TEST_KEY"
        mock_config.headers = {}

        mock_discover.return_value = ["http://10.0.0.1:8000/v1"]

        discovery = WorkerServerDiscovery(mock_config, "my-model")

        with patch.object(discovery, "_setup_clients") as mock_setup:
            mock_client = MagicMock()
            mock_setup.return_value = [mock_client]

            changed = asyncio.run(discovery.refresh())

            assert changed is True
            mock_discover.assert_called_once_with("test.hostname", 8000, "my-model")


def test_worker_server_discovery_refresh_no_change_when_urls_same():
    with (
        patch("prime_rl.utils.elastic.get_logger"),
        patch("prime_rl.utils.elastic.discover_ready_servers") as mock_discover,
    ):
        mock_config = MagicMock()
        mock_config.elastic.hostname = "test.hostname"
        mock_config.elastic.port = 8000
        mock_config.elastic.sync_interval = 0.0
        mock_config.timeout = 30
        mock_config.api_key_var = "TEST_KEY"
        mock_config.headers = {}

        mock_discover.return_value = ["http://10.0.0.1:8000/v1"]

        discovery = WorkerServerDiscovery(mock_config, "my-model")
        discovery._last_urls = {"http://10.0.0.1:8000/v1"}
        discovery._last_refresh = 0  # Allow refresh

        changed = asyncio.run(discovery.refresh())

        assert changed is False


def test_worker_server_discovery_refresh_returns_false_when_no_hostname():
    with patch("prime_rl.utils.elastic.get_logger"):
        mock_config = MagicMock()
        mock_config.elastic.hostname = None
        mock_config.elastic.port = 8000
        mock_config.elastic.sync_interval = 5.0

        discovery = WorkerServerDiscovery(mock_config, "my-model")

        changed = asyncio.run(discovery.refresh())

        assert changed is False


def test_worker_server_discovery_close_clears_clients():
    with patch("prime_rl.utils.elastic.get_logger"):
        mock_config = MagicMock()
        mock_config.elastic.hostname = "test.hostname"
        mock_config.elastic.port = 8000
        mock_config.elastic.sync_interval = 5.0

        discovery = WorkerServerDiscovery(mock_config, "my-model")

        # Add mock clients
        mock_client = AsyncMock()
        discovery._clients = [mock_client]

        asyncio.run(discovery.close())

        assert discovery._clients == []
        mock_client.close.assert_called_once()
