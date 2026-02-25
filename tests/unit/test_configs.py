import sys
from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings, parse_argv

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests that all config files can be loaded by at least one config class."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dummy.py", "@", config_file.as_posix()],
        raising=False,
    )
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            parse_argv(config_cls)
            could_parse.append(True)
        except ValidationError:
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseSettings):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults(monkeypatch):
    """All defaults are applied when no TOML or CLI args are given."""
    monkeypatch.setattr(sys, "argv", ["dummy.py"])
    config = parse_argv(DummyConfig)
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(monkeypatch, tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml")])
    config = parse_argv(DummyConfig)
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(monkeypatch, tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml")])
    config = parse_argv(DummyConfig)
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(monkeypatch, tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml")])
    config = parse_argv(DummyConfig)
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(monkeypatch, tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml")])
    config = parse_argv(DummyConfig)
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults(monkeypatch):
    """CLI args override defaults."""
    monkeypatch.setattr(sys, "argv", ["dummy.py", "--name", "my-run", "--seed", "7"])
    config = parse_argv(DummyConfig)
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_cli(monkeypatch, tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    monkeypatch.setattr(sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml")])
    config = parse_argv(DummyConfig)
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(monkeypatch, tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    monkeypatch.setattr(
        sys, "argv", ["dummy.py", "@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"]
    )
    config = parse_argv(DummyConfig)
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01
