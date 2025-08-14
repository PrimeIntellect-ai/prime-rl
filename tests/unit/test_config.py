import sys
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
from pydantic_settings import BaseSettings

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.validation import (
    validate_shared_async_level,
    validate_shared_ckpt_config,
    validate_shared_max_model_len,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_outputs_dir,
    validate_shared_wandb_config,
)

ConfigType: TypeAlias = Literal["train", "orch", "infer"]

# Map config type to its corresponding settings class
CONFIG_MAP: dict[ConfigType, type[BaseSettings]] = {
    "train": RLTrainerConfig,
    "orch": OrchestratorConfig,
    "infer": InferenceConfig,
}


def get_config_files(config_type: ConfigType) -> list[Path]:
    """Any TOML file inside `configs/`"""
    return list(Path("configs").glob(f"**/{config_type}.toml"))


def get_config_dirs() -> list[Path]:
    """Any directory containing a TOML file inside `configs/`"""
    return sorted(
        {
            cfg_file.parent
            for cfg_type, _ in CONFIG_MAP.items()
            for cfg_file in Path("configs").glob(f"**/{cfg_type}.toml")
        }
    )


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
        ["dummy.py", "@", config_file.as_posix()],
        raising=False,
    )
    config = parse_argv(config_cls)
    assert config is not None


@pytest.mark.parametrize("config_dir", [pytest.param(cfg_dir, id=cfg_dir.as_posix()) for cfg_dir in get_config_dirs()])
def test_shared_config_validation(config_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # Require that all three files are present in the directory
    required = {"train.toml", "orch.toml", "infer.toml"}
    present = {p.name for p in config_dir.glob("*.toml")}
    missing = required - present
    assert not missing, f"Missing required config files in {config_dir}: {', '.join(sorted(missing))}"

    # Load configs
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "train.toml").as_posix()], raising=False)
    trainer = parse_argv(CONFIG_MAP["train"])
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "orch.toml").as_posix()], raising=False)
    orchestrator = parse_argv(CONFIG_MAP["orch"])
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "infer.toml").as_posix()], raising=False)
    inference = parse_argv(CONFIG_MAP["infer"])

    # Validate shared constraints across all three configs
    assert trainer and orchestrator and inference
    validate_shared_ckpt_config(trainer, orchestrator)
    validate_shared_model_name(trainer, orchestrator, inference)
    validate_shared_outputs_dir(trainer, orchestrator)
    validate_shared_wandb_config(trainer, orchestrator)
    validate_shared_max_model_len(orchestrator, inference)
    validate_shared_async_level(trainer, orchestrator)
    validate_shared_max_steps(trainer, orchestrator)
