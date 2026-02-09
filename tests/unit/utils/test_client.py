import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from prime_rl.utils import client as client_module
from prime_rl.utils.client import _is_retryable_lora_error, load_lora_adapter, update_weights


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
    )


def test_load_lora_adapter_retries_on_404_then_succeeds():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 404
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=error_response)
        return success_response

    mock_client.post = mock_post

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert call_count == 2


def test_load_lora_adapter_raises_non_retryable_error_immediately():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 400
    mock_client.post.side_effect = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=error_response)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert exc_info.value.response.status_code == 400
    assert mock_client.post.call_count == 1


def test_update_weights_with_lora_and_adapter_only_checkpoint(monkeypatch, tmp_path):
    mock_client = AsyncMock()
    (tmp_path / "adapter_model.safetensors").touch()
    load_lora_adapter_mock = AsyncMock()
    monkeypatch.setattr(client_module, "load_lora_adapter", load_lora_adapter_mock)

    asyncio.run(update_weights([mock_client], tmp_path, lora_name="test-lora"))

    assert not mock_client.post.called
    load_lora_adapter_mock.assert_awaited_once_with([mock_client], "test-lora", tmp_path)


def test_update_weights_with_lora_and_base_weights_checkpoint(monkeypatch, tmp_path):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "adapter_model.safetensors").touch()
    load_lora_adapter_mock = AsyncMock()
    monkeypatch.setattr(client_module, "load_lora_adapter", load_lora_adapter_mock)

    asyncio.run(update_weights([mock_client], tmp_path, lora_name="test-lora"))

    mock_client.post.assert_called_once_with("/update_weights", json={"weight_dir": tmp_path.as_posix()})
    assert (tmp_path / "NCCL_READY").exists()
    load_lora_adapter_mock.assert_awaited_once_with([mock_client], "test-lora", tmp_path)


def test_update_weights_with_lora_and_base_update_marker(monkeypatch, tmp_path):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    (tmp_path / "BASE_UPDATE_REQUIRED").touch()
    load_lora_adapter_mock = AsyncMock()
    monkeypatch.setattr(client_module, "load_lora_adapter", load_lora_adapter_mock)

    asyncio.run(update_weights([mock_client], tmp_path, lora_name="test-lora"))

    mock_client.post.assert_called_once_with("/update_weights", json={"weight_dir": tmp_path.as_posix()})
    assert (tmp_path / "NCCL_READY").exists()
    load_lora_adapter_mock.assert_awaited_once_with([mock_client], "test-lora", tmp_path)
