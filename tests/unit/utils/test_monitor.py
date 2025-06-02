import json
import os
from unittest.mock import MagicMock, patch

import pytest

from zeroband.utils.monitor import (
    APIMonitor,
    APIMonitorConfig,
    FileMonitor,
    FileMonitorConfig,
    SocketMonitor,
    SocketMonitorConfig,
)


def test_invalid_file_monitor_config():
    with pytest.raises(AssertionError):
        FileMonitor(FileMonitorConfig(enable=True))

    with pytest.raises(AssertionError):
        FileMonitor(FileMonitorConfig(enable=True, path=None))


def test_invalid_socket_monitor_config():
    with pytest.raises(AssertionError):
        SocketMonitor(SocketMonitorConfig(enable=True))

    with pytest.raises(AssertionError):
        SocketMonitor(SocketMonitorConfig(enable=True, path=None))


def test_invalid_api_monitor_config():
    with pytest.raises(AssertionError):
        APIMonitor(APIMonitorConfig(enable=True))

    with pytest.raises(AssertionError):
        APIMonitor(APIMonitorConfig(enable=True, url=None))

    with pytest.raises(AssertionError):
        APIMonitor(APIMonitorConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token=None))


def test_valid_file_monitor_config(tmp_path):
    file_path = (tmp_path / "file_monitor.jsonl").as_posix()
    output = FileMonitor(FileMonitorConfig(enable=True, path=file_path))
    assert output is not None
    assert output.file_path == file_path
    assert output.config.enable


def test_valid_socket_monitor_config(tmp_path):
    socket_path = (tmp_path / "socket_monitor.sock").as_posix()
    output = SocketMonitor(SocketMonitorConfig(enable=True, path=socket_path))
    assert output is not None
    assert output.socket_path == socket_path
    assert output.config.enable


def test_valid_socket_monitor_config_with_env(tmp_path):
    socket_path = (tmp_path / "socket_monitor.sock").as_posix()
    os.environ["PRIME_SOCKET_PATH"] = (tmp_path / "socket_monitor.sock").as_posix()
    output = SocketMonitor(SocketMonitorConfig(enable=True))
    assert output is not None
    assert output.socket_path == socket_path
    assert output.config.enable


def test_valid_http_monitor_config():
    url = "http://localhost:8000/api/v1/metrics"
    auth_token = "test_token"
    os.environ["PRIME_API_URL"] = url
    os.environ["PRIME_API_AUTH_TOKEN"] = auth_token
    output = APIMonitor(APIMonitorConfig(enable=True))
    assert output is not None
    assert output.url == url
    assert output.auth_token == auth_token
    assert output.config.enable


def test_file_monitor(tmp_path):
    # Create file output
    file_path = (tmp_path / "file_monitor.jsonl").as_posix()
    output = FileMonitor(FileMonitorConfig(enable=True, path=file_path))
    assert output is not None
    assert output.config.path == file_path
    assert output.config.enable

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    output.log(test_metrics)

    # Verify the metrics were logged
    with open(file_path, "r") as f:
        assert f.read().strip() == json.dumps(test_metrics)


@pytest.fixture
def mock_socket():
    """Fixture that provides a mocked socket for testing."""
    with patch("socket.socket") as mock_socket_class:
        mock_socket_instance = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket_instance
        yield mock_socket_instance


def test_socket_monitor(mock_socket):
    # Set task ID in environment
    test_task_id = "test-task-123"
    os.environ["PRIME_TASK_ID"] = test_task_id

    # Create socket output
    output = SocketMonitor(SocketMonitorConfig(enable=True, path="/test/socket.sock"))

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    output.log(test_metrics)

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
    output = APIMonitor(APIMonitorConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token="test_token"))

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    output.log(test_metrics)

    assert mock_socket.connect.called_once
    assert mock_socket.sendall.called

    # Get the data that was sent
    sent_data = mock_socket.sendall.call_args[0][0].decode("utf-8")
    assert sent_data.strip() == json.dumps(test_metrics)
