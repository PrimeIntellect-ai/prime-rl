import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl_config import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    RLTrainerConfig,
    SFTTrainerConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests that all config files can be loaded by at least one config class."""
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


BASE_RL_CONFIG = "examples/reverse_text/rl.toml"


class TestSeqLenOverride:
    """Tests that explicit trainer.model.seq_len takes precedence over the shared top-level seq_len."""

    def test_shared_seq_len_propagates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["dummy.py", "@", BASE_RL_CONFIG, "--seq-len", "4096"])
        config = parse_argv(RLConfig)
        assert config.trainer.model.seq_len == 4096
        assert config.orchestrator.seq_len == 4096

    def test_trainer_seq_len_overrides_shared_larger(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys, "argv", ["dummy.py", "@", BASE_RL_CONFIG, "--seq-len", "4096", "--trainer.model.seq-len", "8192"]
        )
        config = parse_argv(RLConfig)
        assert config.trainer.model.seq_len == 8192
        assert config.orchestrator.seq_len == 4096

    def test_trainer_seq_len_below_orchestrator_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys, "argv", ["dummy.py", "@", BASE_RL_CONFIG, "--seq-len", "4096", "--trainer.model.seq-len", "1024"]
        )
        with pytest.raises(ValidationError, match="must be >="):
            parse_argv(RLConfig)

    def test_shared_seq_len_overrides_orchestrator(self, monkeypatch: pytest.MonkeyPatch):
        """Shared seq_len always controls the orchestrator, even if orchestrator.seq_len is set independently."""
        monkeypatch.setattr(
            sys, "argv", ["dummy.py", "@", BASE_RL_CONFIG, "--seq-len", "4096", "--orchestrator.seq-len", "1024"]
        )
        config = parse_argv(RLConfig)
        assert config.orchestrator.seq_len == 4096
