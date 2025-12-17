import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv

# All config config classes
CONFIG_CLASSES = [RLConfig, RLTrainerConfig, SFTTrainerConfig, OrchestratorConfig, InferenceConfig, OfflineEvalConfig]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests that all config files can be loaded by at least one config class."""
    if "intellect_3/evals" in config_file.as_posix():
        pytest.skip("Skipped because uses partial configs, which are not supported by this test.")

    monkeypatch.setattr(
        sys,
        "argv",
        ["dummy.py", "@", config_file.as_posix()],
        raising=False,
    )
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            parse_argv(config_cls)
            could_parse.append(True)
        except ValidationError:
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class TestModelDefaults:
    """Tests for per-model default configuration."""

    def test_model_defaults_applied_when_user_does_not_set(self):
        """Model defaults should be applied when user doesn't explicitly set values."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
            trainer={},
            orchestrator={},
        )
        assert config.orchestrator.trajectory_strategy == "interleaved"
        assert config.inference is not None
        assert config.inference.model.enable_auto_tool_choice is True
        assert config.inference.model.tool_call_parser == "hermes"

    def test_user_config_overrides_model_defaults(self):
        """User-provided config should override model defaults."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
            trainer={},
            orchestrator={"trajectory_strategy": "branching"},
        )
        assert config.orchestrator.trajectory_strategy == "branching"

    def test_inference_created_from_defaults_when_not_provided(self):
        """Inference config should be created from defaults when user doesn't provide it."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
            trainer={},
            orchestrator={},
        )
        assert config.inference is not None
        assert config.inference.model.enable_auto_tool_choice is True

    def test_inference_defaults_applied_to_user_provided_inference(self):
        """Model defaults should be applied to user-provided inference config for unset fields."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
            trainer={},
            orchestrator={},
            inference={"gpu_memory_utilization": 0.8},
        )
        assert config.inference is not None
        assert config.inference.gpu_memory_utilization == 0.8
        assert config.inference.model.enable_auto_tool_choice is True

    def test_user_inference_model_overrides_defaults(self):
        """User-provided inference.model settings should override model defaults."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
            trainer={},
            orchestrator={},
            inference={"model": {"enable_auto_tool_choice": False}},
        )
        assert config.inference is not None
        assert config.inference.model.enable_auto_tool_choice is False

    def test_no_defaults_for_unknown_model(self):
        """No defaults should be applied for models not in MODEL_DEFAULTS."""
        config = RLConfig(
            model={"name": "some/unknown-model"},
            trainer={},
            orchestrator={},
        )
        assert config.inference is None
        assert config.orchestrator.trajectory_strategy == "interleaved"  # global default

    def test_trainer_impl_default_applied(self):
        """Trainer model impl default should be applied for MoE models."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
            trainer={},
            orchestrator={},
        )
        assert config.trainer.model.impl == "custom"

    def test_trainer_impl_user_override(self):
        """User can override trainer model impl."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
            trainer={"model": {"impl": "hf"}},
            orchestrator={},
        )
        assert config.trainer.model.impl == "hf"

    def test_thinking_model_uses_branching(self):
        """Thinking models should default to branching trajectory strategy."""
        config = RLConfig(
            model={"name": "Qwen/Qwen3-4B-Thinking-2507"},
            trainer={},
            orchestrator={},
        )
        assert config.orchestrator.trajectory_strategy == "branching"
