import sys
from pathlib import Path

import pytest

from prime_rl.eval.config import EvalConfig
from prime_rl.utils.pydantic_config import parse_argv


def get_all_toml_files(directory) -> list[str]:
    config_files = list(Path(directory).glob("**/*.toml"))
    return [file.as_posix() for file in config_files]


@pytest.mark.parametrize("config_file", get_all_toml_files("configs/eval"))
def test_load_eval_configs(config_file: str):
    sys.argv = ["@" + config_file]
    config = parse_argv(EvalConfig)
    assert config is not None
