from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import (
    DisaggregatedInferenceDeploymentConfig,
    InferenceConfig,
    MultiNodeInferenceDeploymentConfig,
)
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli

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
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
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


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_single_node_inference_defaults_to_router_frontend():
    config = InferenceConfig()
    assert config.deployment.type == "single_node"
    assert config.server.port == 8000
    assert config.deployment.router.port == 8000
    assert config.deployment.router.policy == "consistent_hash"
    assert config.deployment.backend_port == 8100


def test_single_node_inference_custom_public_port_updates_router_port():
    config = InferenceConfig.model_validate({"server": {"port": 9000}})
    assert config.server.port == 9000
    assert config.deployment.router.port == 9000
    assert config.deployment.backend_port == 9100


def test_single_node_inference_rejects_mismatched_public_and_router_ports():
    with pytest.raises(ValidationError, match="must match deployment.router.port"):
        InferenceConfig.model_validate(
            {
                "server": {"port": 9000},
                "deployment": {"type": "single_node", "router": {"port": 9001}},
            }
        )


def test_single_node_inference_accepts_flat_router_fields_for_backwards_compatibility():
    config = InferenceConfig.model_validate(
        {
            "deployment": {
                "type": "single_node",
                "router_port": 9000,
                "router_policy": "round_robin",
            }
        }
    )

    assert config.deployment.router.port == 9000
    assert config.deployment.router.policy == "round_robin"


def test_multi_node_inference_router_defaults_port_when_only_policy_is_set():
    deployment = MultiNodeInferenceDeploymentConfig.model_validate({"router": {"policy": "round_robin"}})

    assert deployment.router.port == 8000
    assert deployment.router.policy == "round_robin"


def test_disaggregated_inference_router_defaults_port_when_only_policy_is_set():
    deployment = DisaggregatedInferenceDeploymentConfig.model_validate({"router": {"policy": "round_robin"}})

    assert deployment.router.port == 8000
    assert deployment.router.policy == "round_robin"


def test_rl_config_auto_sets_single_node_router_and_admin_urls():
    config = cli(
        RLConfig,
        args=["@", "configs/ci/integration/rl/start.toml", "--inference.server.port", "9000"],
    )
    assert config.orchestrator.client.base_url == ["http://localhost:9000/v1"]
    assert config.orchestrator.client.admin_base_url == ["http://localhost:9100/v1"]


def test_rl_config_auto_sets_non_conflicting_teacher_inference_ports():
    config = cli(
        RLConfig,
        args=[
            "@",
            "configs/ci/integration/rl/start.toml",
            "--inference.server.port",
            "9000",
            "--deployment.num_teacher_gpus",
            "1",
        ],
    )

    assert config.teacher_inference is not None
    assert config.teacher_inference.server.port == 9001
    assert config.teacher_inference.deployment.router.port == 9001
    assert config.teacher_inference.deployment.backend_port == 9101
    assert config.orchestrator.teacher_model is not None
    assert config.orchestrator.teacher_model.client.base_url == ["http://localhost:9001/v1"]


def test_rl_config_rejects_teacher_inference_backend_port_collisions():
    with pytest.raises(ValidationError, match="must not reuse inference router/backend ports"):
        cli(
            RLConfig,
            args=[
                "@",
                "configs/ci/integration/rl/start.toml",
                "--inference.server.port",
                "9000",
                "--deployment.num_teacher_gpus",
                "1",
                "--teacher_inference.server.port",
                "9001",
                "--teacher_inference.deployment.backend_port",
                "9100",
            ],
        )


def test_rl_config_rejects_disaggregated_inference_for_single_node_deployment():
    with pytest.raises(ValidationError, match="single-node RL only supports inference.deployment.type = 'single_node'"):
        cli(
            RLConfig,
            args=[
                "@",
                "configs/ci/integration/rl/start.toml",
                "--inference.deployment.type",
                "disaggregated",
            ],
        )
