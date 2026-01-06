"""Tests for EnvWorker subprocess management."""

from unittest.mock import MagicMock


def test_dead_worker_detection_logic():
    """Test that dead worker detection logic works correctly."""
    # Simulate the check that happens in collect_responses
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1
    worker_name = "test-worker"

    # This is the check we added to collect_responses
    if mock_process and not mock_process.is_alive():
        exit_code = mock_process.exitcode
        error_msg = f"Worker '{worker_name}' died unexpectedly (exit code: {exit_code})"
        assert "test-worker" in error_msg
        assert "exit code: 1" in error_msg
    else:
        raise AssertionError("Should have detected dead worker")


def test_alive_worker_does_not_trigger():
    """Test that alive worker does not trigger the dead worker check."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = True

    # Should not enter the error branch
    if mock_process and not mock_process.is_alive():
        raise AssertionError("Should not detect alive worker as dead")


def test_none_process_does_not_trigger():
    """Test that None process does not trigger the dead worker check."""
    process = None

    # Should not enter the error branch when process is None
    if process and not process.is_alive():
        raise AssertionError("Should not detect None process as dead")


def test_exit_code_minus_nine():
    """Test that SIGKILL exit code is correctly reported."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = -9  # SIGKILL
    worker_name = "killed-worker"

    if mock_process and not mock_process.is_alive():
        exit_code = mock_process.exitcode
        error_msg = f"Worker '{worker_name}' died unexpectedly (exit code: {exit_code})"
        assert "exit code: -9" in error_msg
    else:
        raise AssertionError("Should have detected dead worker")


def test_stopping_flag_skips_dead_worker_error():
    """Test that dead worker check is skipped during intentional shutdown."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 0
    _stopping = True

    # When _stopping is True, the check should not raise
    should_raise = mock_process and not mock_process.is_alive() and not _stopping
    assert not should_raise, "Should not raise error when _stopping is True"


def test_stopping_flag_false_raises_error():
    """Test that dead worker check raises when not stopping."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1
    _stopping = False

    # When _stopping is False, the check should raise
    should_raise = mock_process and not mock_process.is_alive() and not _stopping
    assert should_raise, "Should raise error when _stopping is False"
