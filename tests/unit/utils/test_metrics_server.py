"""Tests for the Prometheus metrics server."""

import socket
import time
import urllib.request
from contextlib import closing

import pytest

from prime_rl.utils.config import MetricsServerConfig
from prime_rl.utils.metrics_server import PROMETHEUS_AVAILABLE, MetricsServer


def find_free_port() -> int:
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class TestMetricsServerConfig:
    def test_default_values(self):
        config = MetricsServerConfig()
        assert config.port == 8000
        assert config.host == "0.0.0.0"

    def test_custom_port(self):
        config = MetricsServerConfig(port=9090)
        assert config.port == 9090

    def test_invalid_port_low(self):
        with pytest.raises(ValueError):
            MetricsServerConfig(port=0)

    def test_invalid_port_high(self):
        with pytest.raises(ValueError):
            MetricsServerConfig(port=65536)


class TestMetricsServer:
    def test_start_stop(self):
        port = find_free_port()
        server = MetricsServer(MetricsServerConfig(port=port))

        server.start()
        assert server._started
        time.sleep(0.1)

        response = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
        assert response.status == 200

        server.stop()
        assert not server._started

    def test_404_on_unknown_path(self):
        port = find_free_port()
        server = MetricsServer(MetricsServerConfig(port=port))
        server.start()
        time.sleep(0.1)

        try:
            urllib.request.urlopen(f"http://localhost:{port}/unknown", timeout=2)
            pytest.fail("Expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        finally:
            server.stop()

    def test_double_start_is_safe(self):
        port = find_free_port()
        server = MetricsServer(MetricsServerConfig(port=port))
        server.start()
        server.start()  # Should not raise
        server.stop()

    def test_update_metrics(self):
        port = find_free_port()
        server = MetricsServer(MetricsServerConfig(port=port))
        server.start()
        time.sleep(0.1)

        server.update(step=42, loss=0.5, throughput=1000.0, grad_norm=1.5, peak_memory_gib=10.0, learning_rate=1e-4)

        response = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
        content = response.read().decode()

        if PROMETHEUS_AVAILABLE:
            assert "rft_trainer_step" in content
            assert "rft_trainer_loss" in content
            assert "rft_trainer_last_step_timestamp_seconds" in content

        server.stop()

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_metrics_values(self):
        port = find_free_port()
        server = MetricsServer(MetricsServerConfig(port=port))
        server.start()
        time.sleep(0.1)

        server.update(step=100, loss=0.123, throughput=5000.0, grad_norm=2.5, peak_memory_gib=16.0, learning_rate=3e-5)

        response = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
        content = response.read().decode()

        assert "rft_trainer_step 100.0" in content
        assert "rft_trainer_loss 0.123" in content

        server.stop()

    def test_isolated_registry(self):
        """Each MetricsServer instance should have its own registry."""
        port1 = find_free_port()
        port2 = find_free_port()

        server1 = MetricsServer(MetricsServerConfig(port=port1))
        server2 = MetricsServer(MetricsServerConfig(port=port2))

        server1.start()
        server2.start()
        time.sleep(0.1)

        # Update only server1
        server1.update(step=999, loss=0.1, throughput=100, grad_norm=1, peak_memory_gib=1, learning_rate=1e-3)

        # server2 should not have server1's values
        if PROMETHEUS_AVAILABLE:
            resp1 = urllib.request.urlopen(f"http://localhost:{port1}/metrics", timeout=2).read().decode()
            resp2 = urllib.request.urlopen(f"http://localhost:{port2}/metrics", timeout=2).read().decode()
            assert "999.0" in resp1
            assert "999.0" not in resp2

        server1.stop()
        server2.stop()

    def test_port_conflict_raises(self):
        port = find_free_port()
        server1 = MetricsServer(MetricsServerConfig(port=port))
        server1.start()
        time.sleep(0.1)

        server2 = MetricsServer(MetricsServerConfig(port=port))
        with pytest.raises(OSError):
            server2.start()

        server1.stop()
