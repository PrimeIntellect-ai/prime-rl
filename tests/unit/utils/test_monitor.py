import json
from unittest.mock import MagicMock, patch

import pytest

from zeroband.utils.monitor import (
    APIOutput,
    APIOutputConfig,
    FileOutput,
    FileOutputConfig,
    SocketOutput,
    SocketOutputConfig,
)


def test_invalid_file_output_config():
    with pytest.raises(AssertionError):
        FileOutput(FileOutputConfig(enable=True))

    with pytest.raises(AssertionError):
        FileOutput(FileOutputConfig(enable=True, path=None))


def test_invalid_socket_output_config():
    with pytest.raises(AssertionError):
        SocketOutput(SocketOutputConfig(enable=True))

    with pytest.raises(AssertionError):
        SocketOutput(SocketOutputConfig(enable=True, path=None))


def test_invalid_api_output_config():
    with pytest.raises(AssertionError):
        APIOutput(APIOutputConfig(enable=True))

    with pytest.raises(AssertionError):
        APIOutput(APIOutputConfig(enable=True, url=None))

    with pytest.raises(AssertionError):
        APIOutput(APIOutputConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token=None))


def test_valid_file_output_config(tmp_path):
    file_path = (tmp_path / "file_monitor.jsonl").as_posix()
    output = FileOutput(FileOutputConfig(enable=True, path=file_path))
    assert output is not None
    assert output.config.path == file_path
    assert output.config.enable


def test_valid_socket_output_config(tmp_path):
    socket_path = (tmp_path / "socket_monitor.sock").as_posix()
    output = SocketOutput(SocketOutputConfig(enable=True, path=socket_path))
    assert output is not None
    assert output.config.path == socket_path
    assert output.config.enable


def test_valid_http_output_config():
    output = APIOutput(APIOutputConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token="test_token"))
    assert output is not None
    assert output.config.url == "http://localhost:8000/api/v1/metrics"
    assert output.config.auth_token == "test_token"
    assert output.config.enable


def test_file_output(tmp_path):
    # Create file output
    file_path = (tmp_path / "file_monitor.jsonl").as_posix()
    output = FileOutput(FileOutputConfig(enable=True, path=file_path))
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


def test_socket_output(mock_socket):
    # Create socket output
    output = SocketOutput(SocketOutputConfig(enable=True, path="/test/socket.sock"))

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    output.log(test_metrics)

    assert mock_socket.connect.called_once
    assert mock_socket.sendall.called

    # Get the data that was sent
    sent_data = mock_socket.sendall.call_args[0][0].decode("utf-8")
    assert sent_data.strip() == json.dumps(test_metrics)


@pytest.fixture
def mock_api():
    """Fixture that provides a mocked API for testing."""
    with patch("aiohttp.ClientSession") as mock_api_class:
        mock_api_instance = MagicMock()
        mock_api_class.return_value.__enter__.return_value = mock_api_instance
        yield mock_api_instance


@pytest.mark.skip(reason="Does not work yet with async context")
def test_api_output(mock_api):
    # Create API output
    output = APIOutput(APIOutputConfig(enable=True, url="http://localhost:8000/api/v1/metrics", auth_token="test_token"))

    # Test logging metrics
    test_metrics = {"step": 1, "loss": 3.14}
    output.log(test_metrics)

    assert mock_socket.connect.called_once
    assert mock_socket.sendall.called

    # Get the data that was sent
    sent_data = mock_socket.sendall.call_args[0][0].decode("utf-8")
    assert sent_data.strip() == json.dumps(test_metrics)
