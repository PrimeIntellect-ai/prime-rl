"""Tests for UsageReporter."""

import os
from unittest.mock import patch

from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.utils.usage_reporter import UsageReporter


class TestUsageReporterInit:
    def test_disabled_when_config_none(self):
        reporter = UsageReporter(None)
        assert not reporter.enabled

    def test_disabled_when_api_key_missing(self):
        config = PrimeMonitorConfig(base_url="http://test", api_key_var="MISSING_VAR")
        with patch.dict(os.environ, {}, clear=True):
            reporter = UsageReporter(config)
            assert not reporter.enabled

    def test_enabled_when_api_key_present(self):
        config = PrimeMonitorConfig(base_url="http://test", api_key_var="TEST_KEY")
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            reporter = UsageReporter(config)
            assert reporter.enabled
            assert reporter.base_url == "http://test"
            reporter.close()


class TestUsageReporterNoOp:
    def test_report_inference_noop(self):
        reporter = UsageReporter(None)
        reporter.report_inference_usage(run_id="r1", step=1, input_tokens=100, output_tokens=200)

    def test_close_noop(self):
        reporter = UsageReporter(None)
        reporter.close()


class TestConfigDefaults:
    def test_reuses_prime_monitor_defaults(self):
        config = PrimeMonitorConfig()
        assert config.base_url == "https://api.primeintellect.ai/api/internal/rft"
        assert config.api_key_var == "PRIME_API_KEY"
