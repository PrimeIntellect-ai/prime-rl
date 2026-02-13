from unittest.mock import MagicMock, patch

import pytest

from prime_rl.utils.config import MLflowConfig, MLflowWithExtrasConfig
from prime_rl.utils.monitor.mlflow import MLflowMonitor


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("DP_RANK", "0")


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
def test_basic_log(mock_log_metrics, mock_log_params, mock_start_run, mock_set_experiment, mock_set_tracking_uri):
    mock_start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

    config = MLflowConfig(tracking_uri="mlruns", experiment_name="test")
    monitor = MLflowMonitor(config=config)

    monitor.log({"loss": 0.5, "step": 1}, step=1)

    mock_set_tracking_uri.assert_called_once_with("mlruns")
    mock_set_experiment.assert_called_once_with("test")
    mock_start_run.assert_called_once()
    mock_log_metrics.assert_called_once_with({"loss": 0.5}, step=1)


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
def test_log_skips_step_key(
    mock_log_metrics, mock_log_params, mock_start_run, mock_set_experiment, mock_set_tracking_uri
):
    mock_start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

    config = MLflowConfig()
    monitor = MLflowMonitor(config=config)

    monitor.log({"loss": 0.5, "accuracy": 0.9, "step": 10}, step=10)

    mock_log_metrics.assert_called_once_with({"loss": 0.5, "accuracy": 0.9}, step=10)


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
def test_log_skips_non_numeric(
    mock_log_metrics, mock_log_params, mock_start_run, mock_set_experiment, mock_set_tracking_uri
):
    mock_start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

    config = MLflowConfig()
    monitor = MLflowMonitor(config=config)

    monitor.log({"loss": 0.5, "name": "test", "flag": True, "step": 1}, step=1)

    mock_log_metrics.assert_called_once_with({"loss": 0.5}, step=1)


def test_non_master_noop(monkeypatch):
    monkeypatch.setenv("RANK", "1")

    config = MLflowConfig()
    monitor = MLflowMonitor(config=config)

    # Should not raise, just no-op
    monitor.log({"loss": 0.5}, step=1)
    assert monitor.history == [{"loss": 0.5}]
    assert not monitor.is_master


def test_disabled_config():
    monitor = MLflowMonitor(config=None)

    monitor.log({"loss": 0.5}, step=1)
    assert monitor.history == [{"loss": 0.5}]
    assert not monitor.enabled


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
def test_log_distributions(
    mock_log_metrics, mock_log_params, mock_start_run, mock_set_experiment, mock_set_tracking_uri
):
    mock_start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

    config = MLflowWithExtrasConfig()
    monitor = MLflowMonitor(config=config)

    distributions = {"rewards": [1.0, 2.0, 3.0], "advantages": [0.1, 0.2, 0.3]}
    monitor.log_distributions(distributions, step=10)

    call_args = mock_log_metrics.call_args
    logged_metrics = call_args[0][0]
    assert "distributions/rewards/mean" in logged_metrics
    assert "distributions/rewards/std" in logged_metrics
    assert "distributions/rewards/min" in logged_metrics
    assert "distributions/rewards/max" in logged_metrics
    assert "distributions/rewards/median" in logged_metrics
    assert logged_metrics["distributions/rewards/mean"] == 2.0
    assert logged_metrics["distributions/rewards/min"] == 1.0
    assert logged_metrics["distributions/rewards/max"] == 3.0


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.end_run")
def test_close(mock_end_run, mock_log_params, mock_start_run, mock_set_experiment, mock_set_tracking_uri):
    mock_start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

    config = MLflowConfig()
    monitor = MLflowMonitor(config=config)
    monitor.close()

    mock_end_run.assert_called_once()
