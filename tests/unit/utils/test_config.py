"""Tests for environment variable injection in prime-rl config classes."""

import os

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import cli


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove any PRIME_RL_* env vars before each test."""
    for key in list(os.environ):
        if key.startswith("PRIME_RL_"):
            monkeypatch.delenv(key, raising=False)


# ── OrchestratorConfig ──────────────────────────────────────────────


def test_orch_env_var_batch_size(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ORCH_BATCH_SIZE", "512")
    config = OrchestratorConfig.model_validate({})
    assert config.batch_size == 512


def test_orch_env_var_nested(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ORCH_STUDENT__MODEL__NAME", "Qwen/Qwen3-0.6B")
    config = OrchestratorConfig.model_validate({})
    assert config.student.model.name == "Qwen/Qwen3-0.6B"


def test_orch_env_var_cli_wins(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ORCH_BATCH_SIZE", "512")
    config = cli(OrchestratorConfig, args=["--batch-size", "256"])
    assert config.batch_size == 256


# ── TrainerConfig ───────────────────────────────────────────────────


def test_trainer_env_var_max_steps(monkeypatch):
    monkeypatch.setenv("PRIME_RL_TRAINER_MAX_STEPS", "1000")
    config = TrainerConfig.model_validate({})
    assert config.max_steps == 1000


# ── InferenceConfig ────────────────────────────────────────────────


def test_infer_env_var(monkeypatch):
    monkeypatch.setenv("PRIME_RL_INFER_MODEL__MAX_MODEL_LEN", "4096")
    config = InferenceConfig.model_validate({})
    assert config.model.max_model_len == 4096


# ── RLConfig ───────────────────────────────────────────────────────


def test_rl_env_var_propagates_to_orchestrator(monkeypatch):
    """Env vars on RLConfig (prefix PRIME_RL_) propagate to sub-configs
    before the auto_setup_shared_configs validator runs."""
    monkeypatch.setenv("PRIME_RL_ORCHESTRATOR__BATCH_SIZE", "512")
    config = RLConfig.model_validate({"trainer": {}, "orchestrator": {}, "inference": {}})
    assert config.orchestrator.batch_size == 512


# ── No prefix leakage ──────────────────────────────────────────────


def test_orch_prefix_does_not_leak_to_trainer(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ORCH_BATCH_SIZE", "999")
    config = TrainerConfig.model_validate({})
    # TrainerConfig has prefix PRIME_RL_TRAINER_; ORCH_ vars shouldn't affect it
    assert config.max_steps == TrainerConfig().max_steps
