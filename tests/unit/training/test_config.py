"""
Tests all of the config file. useful to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

import os
import sys

import pytest

from zeroband.training.config import Config as TrainingConfig
from zeroband.utils.pydantic_config import parse_argv


def get_all_toml_files(directory):
    toml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".toml"):
                toml_files.append(os.path.join(root, file))
    return toml_files


@pytest.mark.parametrize("config_file_path", get_all_toml_files("configs/training"))
def test_load_train_configs(config_file_path):
    sys.argv = ["train.py", "@" + config_file_path]
    config = parse_argv(TrainingConfig)
    assert config is not None
