import sys
from pathlib import Path

import pytest

from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.utils.pydantic_config import parse_argv


def get_all_toml_files(directory) -> list[str]:
    config_files = list(Path(directory).glob("**/*.toml"))
    orchestrator_config_files = [file.as_posix() for file in config_files]
    return orchestrator_config_files


@pytest.mark.parametrize("config_file", get_all_toml_files("configs/orchestrator"))
def test_load_orchestrator_configs(config_file: str):
    sys.argv = ["orchestrator.py", "@", config_file]
    config = parse_argv(OrchestratorConfig)
    assert config is not None
