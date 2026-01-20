import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.utils.config import ClientConfig
from prime_rl.utils.elastic import (
    DesiredAdapterState,
    ElasticInferencePool,
    LoadedAdapter,
    PodState,
    discover_pod_ips,
)

# =============================================================================
# Tests for discover_pod_ips
# =============================================================================


def test_discover_pod_ips_returns_sorted_ips():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.return_value = ("host", [], ["10.0.0.3", "10.0.0.1", "10.0.0.2"])
        ips = discover_pod_ips("my-service.ns.svc.cluster.local")
        assert ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


def test_discover_pod_ips_returns_empty_on_dns_failure():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.side_effect = socket.gaierror
        ips = discover_pod_ips("nonexistent.svc")
        assert ips == []


def test_discover_pod_ips_returns_empty_when_no_pods():
    with patch("socket.gethostbyname_ex") as mock_dns:
        mock_dns.return_value = ("host", [], [])
        ips = discover_pod_ips("empty-service.svc")
        assert ips == []


# =============================================================================
# Tests for LoadedAdapter
# =============================================================================


def test_loaded_adapter_dataclass():
    adapter = LoadedAdapter(name="my-lora", path=Path("/weights/step_100"), step=100)
    assert adapter.name == "my-lora"
    assert adapter.path == Path("/weights/step_100")
    assert adapter.step == 100


# =============================================================================
# Tests for PodState
# =============================================================================


def test_pod_state_defaults():
    pod = PodState(ip="10.0.0.1", url="http://10.0.0.1:8000")
    assert pod.status == "discovering"
    assert pod.loaded_adapter is None
    assert pod.sync_failures == 0


# =============================================================================
# Tests for ElasticInferencePool
# =============================================================================


@pytest.fixture
def client_config():
    return ClientConfig(base_url=["http://localhost:8000/v1"], timeout=60)


@pytest.fixture
def pool(client_config):
    return ElasticInferencePool(
        headless_service="inference-headless.ns.svc.cluster.local",
        client_config=client_config,
        base_model="Qwen/Qwen2-0.5B",
        port=8000,
        sync_interval=5.0,
    )


class TestElasticInferencePoolInit:
    def test_init_sets_attributes(self, pool):
        assert pool.headless_service == "inference-headless.ns.svc.cluster.local"
        assert pool.base_model == "Qwen/Qwen2-0.5B"
        assert pool.port == 8000
        assert pool.sync_interval == 5.0
        assert pool.num_pods == 0
        assert pool.num_ready_pods == 0
        assert pool.ready_urls == []

    def test_build_url(self, pool):
        assert pool._build_url("10.0.0.1") == "http://10.0.0.1:8000"

    def test_build_inference_url(self, pool):
        assert pool._build_inference_url("10.0.0.1") == "http://10.0.0.1:8000/v1"


class TestAdapterMatching:
    def test_no_adapter_desired_always_matches(self, pool):
        pool._desired = DesiredAdapterState(lora_name=None, lora_path=None, step=0)
        assert pool._adapter_matches_desired(None) is True
        assert pool._adapter_matches_desired(LoadedAdapter("x", Path("/x"), 1)) is True

    def test_adapter_desired_but_none_loaded(self, pool):
        pool._desired = DesiredAdapterState(
            lora_name="my-lora", lora_path=Path("/weights/step_100"), step=100
        )
        assert pool._adapter_matches_desired(None) is False

    def test_adapter_matches_by_path(self, pool):
        pool._desired = DesiredAdapterState(
            lora_name="my-lora", lora_path=Path("/weights/step_100"), step=100
        )
        loaded = LoadedAdapter(name="my-lora", path=Path("/weights/step_100"), step=100)
        assert pool._adapter_matches_desired(loaded) is True

    def test_adapter_matches_by_step(self, pool):
        pool._desired = DesiredAdapterState(
            lora_name="my-lora", lora_path=Path("/weights/step_100"), step=100
        )
        # Different path but same step
        loaded = LoadedAdapter(name="my-lora", path=Path("/other/step_100"), step=100)
        assert pool._adapter_matches_desired(loaded) is True

    def test_adapter_does_not_match(self, pool):
        pool._desired = DesiredAdapterState(
            lora_name="my-lora", lora_path=Path("/weights/step_100"), step=100
        )
        loaded = LoadedAdapter(name="my-lora", path=Path("/weights/step_50"), step=50)
        assert pool._adapter_matches_desired(loaded) is False


class TestReadyUrls:
    def test_ready_urls_returns_only_ready_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
            "10.0.0.2": PodState(ip="10.0.0.2", url="http://10.0.0.2:8000", status="syncing"),
            "10.0.0.3": PodState(ip="10.0.0.3", url="http://10.0.0.3:8000", status="ready"),
        }
        urls = pool.ready_urls
        assert len(urls) == 2
        assert "http://10.0.0.1:8000/v1" in urls
        assert "http://10.0.0.3:8000/v1" in urls

    def test_num_ready_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
            "10.0.0.2": PodState(ip="10.0.0.2", url="http://10.0.0.2:8000", status="unhealthy"),
        }
        assert pool.num_ready_pods == 1
        assert pool.num_pods == 2


class TestReadyUrlsCallback:
    def test_callback_called_when_urls_change(self, pool):
        callback = MagicMock()
        pool.on_ready_urls_changed = callback

        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }
        pool._notify_if_ready_urls_changed()

        callback.assert_called_once_with(["http://10.0.0.1:8000/v1"])

    def test_callback_not_called_when_urls_unchanged(self, pool):
        callback = MagicMock()
        pool.on_ready_urls_changed = callback

        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }
        pool._last_ready_urls = ["http://10.0.0.1:8000/v1"]
        pool._notify_if_ready_urls_changed()

        callback.assert_not_called()

    def test_no_callback_set(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }
        # Should not raise
        pool._notify_if_ready_urls_changed()


class TestPodSync:
    @pytest.mark.asyncio
    async def test_sync_discovers_new_pods(self, pool):
        with patch("prime_rl.utils.elastic.discover_pod_ips") as mock_discover:
            mock_discover.return_value = ["10.0.0.1", "10.0.0.2"]

            with patch.object(pool, "_add_pod", new_callable=AsyncMock) as mock_add:
                mock_add.return_value = True
                added, removed = await pool.sync()

        assert added == 2
        assert removed == 0
        assert mock_add.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_removes_gone_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
            "10.0.0.2": PodState(ip="10.0.0.2", url="http://10.0.0.2:8000", status="ready"),
        }

        with patch("prime_rl.utils.elastic.discover_pod_ips") as mock_discover:
            mock_discover.return_value = ["10.0.0.1"]  # 10.0.0.2 is gone

            with patch.object(pool, "_remove_pod", new_callable=AsyncMock) as mock_remove:
                added, removed = await pool.sync()

        assert added == 0
        assert removed == 1
        mock_remove.assert_called_once_with("10.0.0.2")

    @pytest.mark.asyncio
    async def test_sync_resyncs_non_ready_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="syncing"),
        }

        with patch("prime_rl.utils.elastic.discover_pod_ips") as mock_discover:
            mock_discover.return_value = ["10.0.0.1"]

            with patch.object(pool, "_sync_pod_adapter", new_callable=AsyncMock) as mock_sync:
                await pool.sync()

        mock_sync.assert_called_once_with("10.0.0.1")


class TestAddPod:
    @pytest.mark.asyncio
    async def test_add_pod_success(self, pool):
        mock_admin = AsyncMock()

        with patch.object(pool, "_create_admin_client", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_admin

            with patch.object(pool, "_sync_pod_adapter", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = True
                result = await pool._add_pod("10.0.0.1")

        assert result is True
        assert "10.0.0.1" in pool._pods
        assert "10.0.0.1" in pool._admin_clients

    @pytest.mark.asyncio
    async def test_add_pod_failure_cleans_up(self, pool):
        with patch.object(pool, "_create_admin_client", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            result = await pool._add_pod("10.0.0.1")

        assert result is False
        assert "10.0.0.1" not in pool._pods
        assert "10.0.0.1" not in pool._admin_clients


class TestRemovePod:
    @pytest.mark.asyncio
    async def test_remove_pod_cleans_up(self, pool):
        mock_admin = AsyncMock()
        pool._pods["10.0.0.1"] = PodState(ip="10.0.0.1", url="http://10.0.0.1:8000")
        pool._admin_clients["10.0.0.1"] = mock_admin

        await pool._remove_pod("10.0.0.1")

        assert "10.0.0.1" not in pool._pods
        assert "10.0.0.1" not in pool._admin_clients
        mock_admin.aclose.assert_called_once()


class TestUpdateWeights:
    @pytest.mark.asyncio
    async def test_update_weights_sets_desired_state(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }

        with patch.object(pool, "_sync_pod_adapter", new_callable=AsyncMock):
            await pool.update_weights(
                weights_path=Path("/weights/step_100"),
                lora_name="my-lora",
                step=100,
            )

        assert pool._desired.lora_name == "my-lora"
        assert pool._desired.lora_path == Path("/weights/step_100")
        assert pool._desired.step == 100

    @pytest.mark.asyncio
    async def test_update_weights_syncs_all_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
            "10.0.0.2": PodState(ip="10.0.0.2", url="http://10.0.0.2:8000", status="ready"),
        }

        with patch.object(pool, "_sync_pod_adapter", new_callable=AsyncMock) as mock_sync:
            await pool.update_weights(
                weights_path=Path("/weights/step_100"),
                lora_name="my-lora",
                step=100,
            )

        assert mock_sync.call_count == 2

    @pytest.mark.asyncio
    async def test_update_weights_notifies_callback(self, pool):
        callback = MagicMock()
        pool.on_ready_urls_changed = callback
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }

        with patch.object(pool, "_sync_pod_adapter", new_callable=AsyncMock):
            await pool.update_weights(
                weights_path=Path("/weights/step_100"),
                lora_name="my-lora",
                step=100,
            )

        callback.assert_called_once()


class TestGetLoadedAdapter:
    @pytest.mark.asyncio
    async def test_get_loaded_adapter_parses_step_underscore(self, pool):
        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "my-lora",
                    "parent": "Qwen/Qwen2-0.5B",
                    "root": "/weights/step_100",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_admin.get.return_value = mock_response

        pool._admin_clients["10.0.0.1"] = mock_admin

        adapter = await pool._get_loaded_adapter("10.0.0.1")

        assert adapter is not None
        assert adapter.name == "my-lora"
        assert adapter.path == Path("/weights/step_100")
        assert adapter.step == 100

    @pytest.mark.asyncio
    async def test_get_loaded_adapter_parses_step_hyphen(self, pool):
        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "my-lora",
                    "parent": "Qwen/Qwen2-0.5B",
                    "root": "/weights/step-200",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_admin.get.return_value = mock_response

        pool._admin_clients["10.0.0.1"] = mock_admin

        adapter = await pool._get_loaded_adapter("10.0.0.1")

        assert adapter is not None
        assert adapter.step == 200

    @pytest.mark.asyncio
    async def test_get_loaded_adapter_returns_none_when_no_adapter(self, pool):
        mock_admin = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "base-model", "parent": None}]}
        mock_response.raise_for_status = MagicMock()
        mock_admin.get.return_value = mock_response

        pool._admin_clients["10.0.0.1"] = mock_admin

        adapter = await pool._get_loaded_adapter("10.0.0.1")

        assert adapter is None

    @pytest.mark.asyncio
    async def test_get_loaded_adapter_returns_none_on_error(self, pool):
        mock_admin = AsyncMock()
        mock_admin.get.side_effect = Exception("Connection failed")

        pool._admin_clients["10.0.0.1"] = mock_admin

        adapter = await pool._get_loaded_adapter("10.0.0.1")

        assert adapter is None


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_performs_initial_sync(self, pool):
        with patch.object(pool, "sync", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = (0, 0)
            await pool.start()

        mock_sync.assert_called_once()
        assert pool._started is True
        assert pool._sync_task is not None

        # Cleanup
        await pool.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, pool):
        with patch.object(pool, "sync", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = (0, 0)
            await pool.start()
            await pool.start()  # Second call should be no-op

        mock_sync.assert_called_once()

        # Cleanup
        await pool.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_sync_task(self, pool):
        with patch.object(pool, "sync", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = (0, 0)
            await pool.start()

        assert pool._sync_task is not None
        await pool.stop()

        assert pool._started is False


class TestGetMetrics:
    def test_get_metrics_returns_correct_values(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
            "10.0.0.2": PodState(ip="10.0.0.2", url="http://10.0.0.2:8000", status="syncing"),
        }
        pool._desired = DesiredAdapterState(lora_name="my-lora", lora_path=Path("/w"), step=100)

        metrics = pool.get_metrics()

        assert metrics["elastic/num_pods"] == 2
        assert metrics["elastic/num_ready_pods"] == 1
        assert metrics["elastic/desired_step"] == 100


class TestWaitForReady:
    @pytest.mark.asyncio
    async def test_wait_for_ready_returns_when_enough_pods(self, pool):
        pool._pods = {
            "10.0.0.1": PodState(ip="10.0.0.1", url="http://10.0.0.1:8000", status="ready"),
        }

        with patch.object(pool, "sync", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = (0, 0)
            await pool.wait_for_ready(min_pods=1, timeout=1.0)

        # Should return without timeout

    @pytest.mark.asyncio
    async def test_wait_for_ready_times_out(self, pool):
        pool._pods = {}

        with patch.object(pool, "sync", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = (0, 0)

            with pytest.raises(TimeoutError) as exc_info:
                await pool.wait_for_ready(min_pods=1, timeout=0.1)

        assert "Timed out" in str(exc_info.value)
