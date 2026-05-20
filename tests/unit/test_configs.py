from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.baselines.config import load_config as load_baseline_config
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import EvalConfig, OrchestratorConfig, TrainSamplingConfig
from prime_rl.configs.orchestrator import NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.validation import validate_shared_weight_broadcast

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
    if config_file.parts[:2] == ("configs", "baselines"):
        load_baseline_config(config_file)
        return
    if config_file.parts[:2] == ("configs", "evals"):
        pytest.skip("eval suite TOMLs are consumed by eval runners, not PrimeRL config classes")

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


def test_inference_fp32_lm_head_threads_through_vllm_additional_config():
    config = InferenceConfig(enable_fp32_lm_head=True)

    namespace = config.to_vllm()

    assert namespace.additional_config == {"fp32_lm_head": True}


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_train_sampling_accepts_top_p():
    sampling = TrainSamplingConfig(top_p=0.95)

    assert sampling.to_sampling_args()["top_p"] == 0.95


def test_eval_seed_inherits_to_env():
    config = EvalConfig(num_examples=100, seed=42, env=[{"id": "test-env"}])

    assert config.env[0].num_examples == 100
    assert config.env[0].seed == 42


def test_eval_env_seed_override_wins():
    config = EvalConfig(seed=42, env=[{"id": "test-env", "seed": 7}])

    assert config.env[0].seed == 7


def test_validate_shared_weight_broadcast_rejects_inference_mismatch():
    trainer = TrainerConfig(weight_broadcast=TrainerNCCLWeightBroadcastConfig())
    orchestrator = OrchestratorConfig(weight_broadcast=OrchestratorNCCLWeightBroadcastConfig())
    inference = InferenceConfig()

    with pytest.raises(ValueError, match="inference=filesystem"):
        validate_shared_weight_broadcast(trainer, orchestrator, inference)


def test_gpu_layout_deployment_sets_one_gpu_inference_pool(tmp_path):
    config_path = tmp_path / "gpu_layout.toml"
    write_toml(
        config_path,
        {
            "max_steps": 1,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "weight_broadcast": {"type": "nccl"},
            "trainer": {},
            "orchestrator": {"client": {"dp_rank_count": 6}},
            "deployment": {
                "type": "gpu_layout",
                "gpus_per_node": 4,
                "nodes": [
                    {"inference": [0, 1, 2, 3]},
                    {"inference": [0, 1], "trainer": [2, 3]},
                ],
            },
            "inference": {"parallel": {"tp": 1, "dp": 1}},
        },
    )

    config = cli(RLConfig, args=["@", config_path.as_posix()])

    assert config.deployment.total_infer_gpus == 6
    assert config.deployment.total_train_gpus == 2
    assert config.orchestrator.client.dp_rank_count == 1
    assert config.orchestrator.num_train_workers == 2
    assert config.inference.parallel.dp == 1
    assert config.inference.data_parallel_size_local == 1
    assert config.inference.api_server_count == 1
    assert config.trainer.weight_broadcast.inference_world_size == 6
    assert config.orchestrator.weight_broadcast.inference_world_size == 6
    assert config.trainer.weight_broadcast.host == "0.0.0.0"


def test_gpu_layout_rejects_overlapping_gpu_roles(tmp_path):
    config_path = tmp_path / "gpu_layout_bad.toml"
    write_toml(
        config_path,
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {},
            "deployment": {
                "type": "gpu_layout",
                "gpus_per_node": 4,
                "nodes": [{"inference": [0], "trainer": [0]}],
            },
        },
    )

    with pytest.raises(ConfigFileError, match="assigned to both inference and trainer"):
        cli(RLConfig, args=["@", config_path.as_posix()])


def test_gpu_layout_rejects_layout_without_inference_gpus(tmp_path):
    config_path = tmp_path / "gpu_layout_no_infer.toml"
    write_toml(
        config_path,
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {},
            "deployment": {
                "type": "gpu_layout",
                "gpus_per_node": 4,
                "nodes": [{"trainer": [0, 1]}],
            },
        },
    )

    with pytest.raises(ConfigFileError, match="at least one inference GPU"):
        cli(RLConfig, args=["@", config_path.as_posix()])
