"""Tests for the Prometheus metrics server."""

import socket
import time
import urllib.request
from contextlib import closing

import pytest
from prometheus_client import CollectorRegistry, Gauge

from prime_rl.configs.shared import MetricsServerConfig
from prime_rl.utils.metrics_server import HealthServer, MetricsServer


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
    registry = CollectorRegistry()
    Gauge("test_metric", "Test metric", registry=registry).set(1)
    server = MetricsServer(MetricsServerConfig(port=port), registry=registry)

    server.start()
    assert server._started
    time.sleep(0.1)

    response = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
    assert response.status == 200
    content = response.read().decode()
    assert "test_metric" in content

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
    registry = CollectorRegistry()
    server = MetricsServer(MetricsServerConfig(port=port), registry=registry)
    server.start()
    server.start()  # Should not raise
    server.stop()


def test_server_isolated_registry():
    """Each MetricsServer instance should have its own registry."""
    port1 = find_free_port()
    port2 = find_free_port()

    server1 = MetricsServer(MetricsServerConfig(port=port1))
    server2 = MetricsServer(MetricsServerConfig(port=port2))
    Gauge("test_metric", "Test metric", registry=server1.registry).set(1)
    Gauge("test_metric", "Test metric", registry=server2.registry).set(2)

    server1.start()
    server2.start()
    time.sleep(0.1)

    resp1 = urllib.request.urlopen(f"http://localhost:{port1}/metrics", timeout=2).read().decode()
    resp2 = urllib.request.urlopen(f"http://localhost:{port2}/metrics", timeout=2).read().decode()
    assert "test_metric 1.0" in resp1
    assert "test_metric 1.0" not in resp2
    assert "test_metric 2.0" in resp2

    server1.stop()
    server2.stop()


def test_server_port_conflict_raises():
    port = find_free_port()
    registry1 = CollectorRegistry()
    server1 = MetricsServer(MetricsServerConfig(port=port), registry=registry1)
    server1.start()
    time.sleep(0.1)

    registry2 = CollectorRegistry()
    server2 = MetricsServer(MetricsServerConfig(port=port), registry=registry2)
    with pytest.raises(OSError):
        server2.start()

    server1.stop()


def test_health_server_start_stop():
    port = find_free_port()
    server = HealthServer(port=port)

    server.start()
    assert server._started
    time.sleep(0.1)

    response = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
    assert response.status == 200
    assert response.read() == b"ok\n"

    server.stop()
    assert not server._started


def test_health_server_returns_404_on_metrics():
    """HealthServer should not expose /metrics endpoint."""
    port = find_free_port()
    server = HealthServer(port=port)
    server.start()
    time.sleep(0.1)

    try:
        urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
        pytest.fail("Expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    finally:
        server.stop()


def test_metrics_server_health_endpoint():
    """MetricsServer should also expose /health endpoint."""
    port = find_free_port()
    server = MetricsServer(MetricsServerConfig(port=port))
    server.start()
    time.sleep(0.1)

    response = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
    assert response.status == 200
    assert response.read() == b"ok\n"

    server.stop()
