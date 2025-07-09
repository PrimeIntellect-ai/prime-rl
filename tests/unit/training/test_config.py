import sys
from pathlib import Path

import pytest

from zeroband.trainer.config import TrainingConfig
from zeroband.utils.pydantic_config import parse_argv


def get_all_toml_files(directory) -> list[str]:
    config_files = list(Path(directory).glob("**/*.toml"))
    train_config_files = [file.as_posix() for file in config_files if "orchestrator" not in file.as_posix()]
    return train_config_files


@pytest.mark.parametrize("config_file", get_all_toml_files("configs/training"))
def test_load_train_configs(config_file: str):
    sys.argv = ["train.py", "@" + config_file]
    config = parse_argv(TrainingConfig)
    assert config is not None
