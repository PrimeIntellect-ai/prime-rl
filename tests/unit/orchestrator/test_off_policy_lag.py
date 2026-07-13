"""Tests for target_lag orchestrator config (#2812)."""

import pytest
from pydantic import ValidationError

from prime_rl.configs.orchestrator import OrchestratorConfig


def test_target_lag_defaults_to_one():
    cfg = OrchestratorConfig()
    assert cfg.target_lag == 1


def test_target_lag_rejects_zero():
    with pytest.raises(ValidationError):
        OrchestratorConfig(target_lag=0)
