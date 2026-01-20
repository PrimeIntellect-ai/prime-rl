import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from prime_rl.utils.elastic import (
    ElasticInferencePool,
    LoadedAdapter,
    check_server_model,
    discover_ready_servers,
    discover_server_ips,
)


class TestDiscoverServerIps:
    def test_returns_sorted_ips(self):
        with patch("socket.gethostbyname_ex") as mock_dns:
            mock_dns.return_value = ("hostname", [], ["10.0.0.3", "10.0.0.1", "10.0.0.2"])
            result = discover_server_ips("test.hostname")
            assert result == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_returns_empty_list_on_dns_failure(self):
        with patch("socket.gethostbyname_ex") as mock_dns:
            mock_dns.side_effect = socket.gaierror("DNS lookup failed")
            result = discover_server_ips("nonexistent.hostname")
            assert result == []

    def test_returns_single_ip(self):
        with patch("socket.gethostbyname_ex") as mock_dns:
            mock_dns.return_value = ("hostname", [], ["10.0.0.1"])
            result = discover_server_ips("single.hostname")
            assert result == ["10.0.0.1"]


class TestCheckServerModel:
    def test_returns_true_when_model_found(self):
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

    def test_returns_false_when_model_not_found(self):
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

    def test_returns_false_on_connection_error(self):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")

            has_model, is_healthy = asyncio.run(check_server_model("http://10.0.0.1:8000", "model-a"))

            assert has_model is False
            assert is_healthy is False

    def test_returns_false_on_http_error(self):
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


class TestDiscoverReadyServers:
    def test_returns_servers_with_model_when_any_have_it(self):
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

    def test_returns_all_healthy_when_none_have_model(self):
        with (
            patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover,
            patch("prime_rl.utils.elastic.check_server_model") as mock_check,
        ):
            mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]
            mock_check.return_value = (False, True)  # no model, but healthy

            result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

            assert set(result) == {"http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"}

    def test_returns_empty_when_no_dns_records(self):
        with patch("prime_rl.utils.elastic.discover_server_ips") as mock_discover:
            mock_discover.return_value = []

            result = asyncio.run(discover_ready_servers("test.hostname", 8000, "my-lora"))

            assert result == []

    def test_excludes_unhealthy_servers(self):
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


class TestLoadedAdapter:
    def test_creation(self):
        adapter = LoadedAdapter(name="my-lora", path=Path("/weights/step_100"), step=100)
        assert adapter.name == "my-lora"
        assert adapter.path == Path("/weights/step_100")
        assert adapter.step == 100


class TestElasticInferencePoolAdapterMatching:
    def test_adapter_matches_when_no_adapter_desired(self):
        with patch("prime_rl.utils.elastic.get_logger"):
            pool = ElasticInferencePool(
                hostname="test.hostname",
                client_config=MagicMock(),
                port=8000,
            )
            # No adapter desired (base model inference)
            assert pool._adapter_matches_desired(None) is True
            assert pool._adapter_matches_desired(LoadedAdapter("x", Path("/x"), 0)) is True

    def test_adapter_matches_by_path(self):
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

    def test_adapter_matches_by_step_when_nonzero(self):
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

    def test_adapter_does_not_match_by_zero_step(self):
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

    def test_returns_false_when_no_adapter_loaded(self):
        with patch("prime_rl.utils.elastic.get_logger"):
            pool = ElasticInferencePool(
                hostname="test.hostname",
                client_config=MagicMock(),
                port=8000,
            )
            pool._desired.lora_path = Path("/weights/step_100")
            pool._desired.step = 100

            assert pool._adapter_matches_desired(None) is False
