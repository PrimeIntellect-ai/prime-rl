import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.config import TokenizerConfig
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


def test_rl_config_updates_tokenizer_name_when_not_explicit():
    trainer = RLTrainerConfig()
    orchestrator = OrchestratorConfig()
    config = RLConfig(
        trainer=trainer,
        orchestrator=orchestrator,
        model={"name": "minpeter/Qwen3-0.6B-Reverse-Text-SFT"},
    )
    assert config.trainer.model.name == "minpeter/Qwen3-0.6B-Reverse-Text-SFT"
    assert config.trainer.tokenizer.name == "minpeter/Qwen3-0.6B-Reverse-Text-SFT"


def test_rl_config_preserves_explicit_tokenizer_name():
    trainer = RLTrainerConfig(tokenizer=TokenizerConfig(name="Qwen/Qwen3-0.6B"))
    orchestrator = OrchestratorConfig()
    config = RLConfig(
        trainer=trainer,
        orchestrator=orchestrator,
        model={"name": "minpeter/Qwen3-0.6B-Reverse-Text-SFT"},
    )
    assert config.trainer.model.name == "minpeter/Qwen3-0.6B-Reverse-Text-SFT"
    assert config.trainer.tokenizer.name == "Qwen/Qwen3-0.6B"
