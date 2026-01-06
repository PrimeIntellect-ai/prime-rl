"""Tests for the Prometheus metrics server."""

import socket
import time
import urllib.request
from contextlib import closing

import pytest

from prime_rl.utils.config import MetricsServerConfig
from prime_rl.utils.metrics_server import PROMETHEUS_AVAILABLE, MetricsServer


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def test_config_default_values():
    config = MetricsServerConfig()
    assert config.port == 8000
    assert config.host == "0.0.0.0"


def test_config_custom_port():
    config = MetricsServerConfig(port=9090)
    assert config.port == 9090


def test_config_invalid_port_low():
    with pytest.raises(ValueError):
        MetricsServerConfig(port=0)


def test_config_invalid_port_high():
    with pytest.raises(ValueError):
        MetricsServerConfig(port=65536)


def test_server_start_stop():
    port = find_free_port()
    server = MetricsServer(MetricsServerConfig(port=port))

    server.start()
    assert server._started
    time.sleep(0.1)

    response = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
    assert response.status == 200

    server.stop()
    assert not server._started


def test_server_returns_404_on_unknown_path():
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


def test_server_double_start_is_safe():
    port = find_free_port()
    server = MetricsServer(MetricsServerConfig(port=port))
    server.start()
    server.start()  # Should not raise
    server.stop()


def test_server_update_metrics():
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
def test_server_metrics_values_are_correct():
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


def test_server_isolated_registry():
    """Each MetricsServer instance should have its own registry."""
    port1 = find_free_port()
    port2 = find_free_port()

    server1 = MetricsServer(MetricsServerConfig(port=port1))
    server2 = MetricsServer(MetricsServerConfig(port=port2))

    server1.start()
    server2.start()
    time.sleep(0.1)

    server1.update(step=999, loss=0.1, throughput=100, grad_norm=1, peak_memory_gib=1, learning_rate=1e-3)

    if PROMETHEUS_AVAILABLE:
        resp1 = urllib.request.urlopen(f"http://localhost:{port1}/metrics", timeout=2).read().decode()
        resp2 = urllib.request.urlopen(f"http://localhost:{port2}/metrics", timeout=2).read().decode()
        assert "999.0" in resp1
        assert "999.0" not in resp2

    server1.stop()
    server2.stop()


def test_server_port_conflict_raises():
    port = find_free_port()
    server1 = MetricsServer(MetricsServerConfig(port=port))
    server1.start()
    time.sleep(0.1)

    server2 = MetricsServer(MetricsServerConfig(port=port))
    with pytest.raises(OSError):
        server2.start()

    server1.stop()
