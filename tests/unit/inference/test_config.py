import sys
from pathlib import Path

import pytest

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def get_all_toml_files(directory) -> list[str]:
    config_files = list(Path(directory).glob("**/infer.toml"))
    return [file.as_posix() for file in config_files]


@pytest.mark.parametrize("config_file", get_all_toml_files("configs/"))
def test_load_inference_configs(config_file: str):
    sys.argv = ["infer.py", "@", config_file]
    config = parse_argv(InferenceConfig)
    assert config is not None
