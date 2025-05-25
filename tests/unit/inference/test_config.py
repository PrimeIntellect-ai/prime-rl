"""
Tests all of the config file. useful to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

import os

import pytest
import tomli
from pydantic import ValidationError

from zeroband.infer import Config as InferenceConfig


def get_all_toml_files(directory):
    toml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".toml"):
                toml_files.append(os.path.join(root, file))
    return toml_files


@pytest.mark.parametrize("config_file_path", get_all_toml_files("configs/inference"))
def test_load_inference_configs(config_file_path):
    with open(f"{config_file_path}", "rb") as f:
        content = tomli.load(f)
    config = InferenceConfig(**content)
    assert config is not None


def test_throw_error_for_dp_and_pp():
    with pytest.raises(ValidationError):
        config = InferenceConfig(**{"dp": 2, "pp": {"world_size": 2}})
        print(config)
