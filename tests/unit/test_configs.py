from pathlib import Path

import pytest
import tomli
from pydantic import ValidationError

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig

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

    with open(config_file, "rb") as f:
        config = tomli.load(f)

    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            config_cls(**config)
            could_parse.append(True)
        except ValidationError:
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"
