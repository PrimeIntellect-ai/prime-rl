import sys
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
from pydantic_settings import BaseSettings

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import TrainerConfig
from prime_rl.utils.pydantic_config import parse_argv

ConfigType: TypeAlias = Literal["train", "orch", "infer"]

# Map config type to its corresponding settings class
CONFIG_MAP: dict[ConfigType, type[BaseSettings]] = {
    "train": TrainerConfig,
    "orch": OrchestratorConfig,
    "infer": InferenceConfig,
}


def get_toml_files(config_type: ConfigType) -> list[Path]:
    return list(Path("configs").glob(f"**/{config_type}.toml"))


@pytest.mark.parametrize(
    "config_cls, config_file",
    [
        pytest.param(
            cfg_cls,
            cfg_file,
            id=f"{cfg_type}::{cfg_file.as_posix()}",
        )
        for cfg_type, cfg_cls in CONFIG_MAP.items()
        for cfg_file in Path("configs").glob(f"**/{cfg_type}.toml")
    ],
)
def test_load_configs(
    config_cls: type[BaseSettings],
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that each individual config file can be loaded into the corresponding config class."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dummy_executable.py", "@", config_file.as_posix()],
        raising=False,
    )
    config = parse_argv(config_cls)
    assert config is not None
