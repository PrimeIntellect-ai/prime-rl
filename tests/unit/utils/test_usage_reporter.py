import os
from unittest.mock import patch

import pytest

from prime_rl.utils.usage_reporter import UsageConfig, UsageReporter


@pytest.fixture
def usage_config():
    return UsageConfig(base_url="http://localhost:8000/api/internal/rft")


def test_disabled_when_no_config():
    reporter = UsageReporter(None)
    assert reporter.is_enabled is False


def test_disabled_when_api_key_missing(usage_config):
    with patch.dict(os.environ, {}, clear=True):
        reporter = UsageReporter(usage_config)
        assert reporter.is_enabled is False


def test_enabled_when_api_key_set(usage_config):
    with patch.dict(os.environ, {"PRIME_API_KEY": "test-key"}):
        reporter = UsageReporter(usage_config)
        assert reporter.is_enabled is True
        reporter.close()


def test_report_inference_noop_when_disabled():
    reporter = UsageReporter(None)
    reporter.report_inference_usage("run1", 0, 100, 200)


def test_report_training_noop_when_disabled():
    reporter = UsageReporter(None)
    reporter.report_training_usage("run1", 0, 5000)


def test_close_noop_when_disabled():
    reporter = UsageReporter(None)
    reporter.close()
