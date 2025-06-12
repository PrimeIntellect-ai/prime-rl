import json
import os
from unittest.mock import MagicMock, patch

import pytest

from zeroband.inference.config import Config as InferenceConfig
from zeroband.utils.monitor import (
    APIMonitor,
    APIMonitorConfig,
    FileMonitor,
    FileMonitorConfig,
    SocketMonitor,
    SocketMonitorConfig,
)


def test_invalid_file_monitor_config():
    with pytest.raises(ValueError):
        FileMonitorConfig(enable=True)

    with pytest.raises(ValueError):
        FileMonitorConfig(enable=True, path=None)


def test_invalid_socket_monitor_config():
    with pytest.raises(ValueError):
        SocketMonitorConfig(enable=True)

    with pytest.raises(ValueError):
        SocketMonitorConfig(enable=True, path=None)


def test_invalid_api_monitor_config():
    with pytest.raises(ValueError):
        APIMonitorConfig(enable=True)

    with pytest.raises(ValueError):
        APIMonitorConfig(enable=True, url=None)

    with pytest.raises(ValueError):
        APIMonitorConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token=None)


def test_valid_file_monitor_config(tmp_path):
    file_path = tmp_path / "file_monitor.jsonl"
    file_monitor_config = FileMonitorConfig(enable=True, path=file_path)
    assert file_monitor_config is not None
    assert file_monitor_config.path == file_path
    assert file_monitor_config.enable


def test_valid_file_monitor_config_with_env(tmp_path):
    file_path = tmp_path / "file_monitor.jsonl"
    os.environ["PRIME_MONITOR__FILE__ENABLE"] = "true"
    os.environ["PRIME_MONITOR__FILE__PATH"] = file_path.as_posix()
    config = InferenceConfig()
    assert config.monitor.file is not None
    assert config.monitor.file.path == file_path
    assert config.monitor.file.enable


def test_valid_socket_monitor_config(tmp_path):
    socket_path = tmp_path / "socket_monitor.sock"
    socket_monitor_config = SocketMonitorConfig(enable=True, path=socket_path)
    assert socket_monitor_config is not None
    assert socket_monitor_config.path == socket_path
    assert socket_monitor_config.enable


def test_valid_socket_monitor_config_with_env(tmp_path):
    socket_path = tmp_path / "socket_monitor.sock"
    os.environ["PRIME_MONITOR__SOCKET__ENABLE"] = "true"
    os.environ["PRIME_MONITOR__SOCKET__PATH"] = socket_path.as_posix()
    config = InferenceConfig()
    assert config.monitor.socket is not None
    assert config.monitor.socket.path == socket_path
    assert config.monitor.socket.enable


def test_valid_api_monitor_config():
    url = "http://localhost:8000/api/v1/metrics"
    auth_token = "test_token"
    api_monitor_config = APIMonitorConfig(enable=True, url=url, auth_token=auth_token)
    assert api_monitor_config is not None
    assert api_monitor_config.url == url
    assert api_monitor_config.auth_token == auth_token
    assert api_monitor_config.enable


def test_valid_api_monitor_config_with_env():
    url = "http://localhost:8000/api/v1/metrics"
    auth_token = "test_token"
    os.environ["PRIME_MONITOR__API__ENABLE"] = "true"
    os.environ["PRIME_MONITOR__API__URL"] = url
    os.environ["PRIME_MONITOR__API__AUTH_TOKEN"] = auth_token
    config = InferenceConfig()
    assert config.monitor.api is not None
    assert config.monitor.api.url == url
    assert config.monitor.api.auth_token == auth_token
    assert config.monitor.api.enable


def test_file_monitor(tmp_path):
    # Create file output
    file_path = tmp_path / "file_monitor.jsonl"
    file_monitor_config = FileMonitorConfig(enable=True, path=file_path)
    file_monitor = FileMonitor(file_monitor_config)
    assert file_monitor_config is not None
    assert file_monitor_config.path == file_path
    assert file_monitor_config.enable

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    file_monitor.log(test_metrics)

    # Verify the metrics were logged
    with open(file_path.as_posix(), "r") as f:
        assert f.read().strip() == json.dumps(test_metrics)


@pytest.fixture
def mock_socket():
    """Fixture that provides a mocked socket for testing."""
    with patch("socket.socket") as mock_socket_class:
        mock_socket_instance = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket_instance
        yield mock_socket_instance


def test_socket_monitor(mock_socket):
    # Create socket output
    test_task_id = "test-task-123"
    socket_monitor_config = SocketMonitorConfig(enable=True, path="/test/socket.sock")
    socket_monitor = SocketMonitor(socket_monitor_config, task_id=test_task_id)

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    socket_monitor.log(test_metrics)

    assert mock_socket.connect.called_once
    assert mock_socket.sendall.called

    # Get the data that was sent
    sent_data = mock_socket.sendall.call_args[0][0].decode("utf-8")
    expected_data = json.dumps({**test_metrics, "task_id": test_task_id})
    assert sent_data.strip() == expected_data


@pytest.fixture
def mock_api():
    """Fixture that provides a mocked API for testing."""
    with patch("aiohttp.ClientSession") as mock_api_class:
        mock_api_instance = MagicMock()
        mock_api_class.return_value.__enter__.return_value = mock_api_instance
        yield mock_api_instance


@pytest.mark.skip(reason="Does not work yet with async context")
def test_api_monitor(mock_api):
    # Create API output
    api_monitor_config = APIMonitorConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token="test_token")
    api_monitor = APIMonitor(api_monitor_config)

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    api_monitor.log(test_metrics)

    assert mock_api.post.called_once
    assert mock_api.post.called_with(api_monitor_config.url, json=test_metrics, headers=api_monitor_config.headers)

    # Get the data that was sent
    sent_data = mock_socket.sendall.call_args[0][0].decode("utf-8")
    assert sent_data.strip() == json.dumps(test_metrics)
