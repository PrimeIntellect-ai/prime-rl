"""
Tests all of the config file. useful to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

import os
import sys

import pytest
from pydantic import ValidationError

from zeroband.inference.config import Config as InferenceConfig
from zeroband.utils.pydantic_config import parse_argv


def get_all_toml_files(directory):
    toml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".toml"):
                toml_files.append(os.path.join(root, file))
    return toml_files


@pytest.mark.parametrize("config_file_path", get_all_toml_files("configs/inference"))
def test_load_inference_configs(config_file_path):
    sys.argv = ["inference.py", "@" + config_file_path]
    config = parse_argv(InferenceConfig)
    assert config is not None


def test_throw_error_for_dp_and_pp():
    with pytest.raises(ValidationError):
        InferenceConfig(**{"parallel": {"dp": 2, "pp": {"world_size": 2}}})
