import sys
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pytest

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.validation import (
    validate_shared_async_level,
    validate_shared_ckpt_config,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_wandb_config,
)

ConfigType: TypeAlias = Literal["rl/train", "sft/train", "orch", "infer", "eval"]

# Map config type to its corresponding settings class
CONFIG_MAP: dict[ConfigType, Any] = {
    "rl/train": RLTrainerConfig,
    "sft/train": SFTTrainerConfig,
    "orch": OrchestratorConfig,
    "infer": InferenceConfig,
    "eval": OfflineEvalConfig,
}


def get_config_files(config_type: ConfigType) -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").glob(f"**/{config_type}.toml"))
    example_files = list(Path("examples").glob(f"**/{config_type}.toml"))
    return config_files + example_files


def test_got_all_config_files():
    found_files = []
    for cfg_type, _ in CONFIG_MAP.items():
        found_files.extend(get_config_files(cfg_type))
    all_files = list(Path("configs").glob("**/*.toml")) + list(Path("examples").glob("**/*.toml"))
    assert len(found_files) == len(all_files), (
        f"Missing {len(all_files) - len(found_files)} config files: {', '.join(sorted(set([f.as_posix() for f in all_files]) - set([f.as_posix() for f in found_files])))}"
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
    config_cls,
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


def get_rl_dirs() -> list[Path]:
    """Any sub-directory named `rl` inside `/configs` or `/examples`"""
    return [config_dir for config_dir in Path("configs").glob("**/rl")] + [
        config_dir for config_dir in Path("examples").glob("**/rl")
    ]


@pytest.mark.parametrize("config_dir", [pytest.param(cfg_dir, id=cfg_dir.as_posix()) for cfg_dir in get_rl_dirs()])
def test_rl_configs(config_dir: Path, monkeypatch: pytest.MonkeyPatch):
    if config_dir.parent.name == "debug":
        pytest.skip("Skipping debug configs")

    # Require that all three files are present in the directory
    required = {"train.toml", "orch.toml", "infer.toml"}
    present = {p.name for p in config_dir.glob("*.toml")}
    missing = required - present
    assert not missing, f"Missing required config files in {config_dir}: {', '.join(sorted(missing))}"

    # Load configs
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "train.toml").as_posix()], raising=False)
    trainer = parse_argv(CONFIG_MAP["rl/train"])
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "orch.toml").as_posix()], raising=False)
    orchestrator = parse_argv(CONFIG_MAP["orch"])
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", (config_dir / "infer.toml").as_posix()], raising=False)
    inference = parse_argv(CONFIG_MAP["infer"])

    # Validate shared constraints across all three configs
    assert trainer and orchestrator and inference
    validate_shared_ckpt_config(trainer, orchestrator)
    validate_shared_model_name(trainer, orchestrator, inference)
    validate_shared_output_dir(trainer, orchestrator)
    validate_shared_wandb_config(trainer, orchestrator)
    validate_shared_async_level(trainer, orchestrator)
    validate_shared_max_steps(trainer, orchestrator)
