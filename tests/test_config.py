"""
Tests all of the config file. usefull to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

import os
from zeroband.train import Config as TrainingConfig
from zeroband.inference import Config as InferenceConfig
import pytest
import tomli


def get_all_toml_files(directory):
    toml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".toml"):
                toml_files.append(os.path.join(root, file))
    return toml_files


config_file_paths = get_all_toml_files("configs/training")
inference_file_paths = get_all_toml_files("configs/inference")


@pytest.mark.parametrize("config_file_path", config_file_paths)
def test_load_config(config_file_path):
    with open(f"{config_file_path}", "rb") as f:
        content = tomli.load(f)
    config = TrainingConfig(**content)
    assert config is not None


@pytest.mark.parametrize("config_file_path", inference_file_paths)
def test_load_inference_config(config_file_path):
    with open(f"{config_file_path}", "rb") as f:
        content = tomli.load(f)
    config = InferenceConfig(**content)
    assert config is not None
