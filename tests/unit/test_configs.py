from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
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


def test_inference_config_translates_sglang_args():
    config = InferenceConfig.model_validate(
        {
            "backend": "sglang",
            "server": {"host": "0.0.0.0", "port": 9000},
            "model": {
                "name": "Qwen/Qwen3-0.6B",
                "max_model_len": 4096,
                "enforce_eager": True,
                "chat_template": "qwen3",
            },
            "parallel": {"tp": 2, "dp": 1},
            "enable_prefix_caching": False,
            "enable_fp32_lm_head": True,
            "gpu_memory_utilization": 0.8,
            "sglang_extra": {"attention_backend": "triton"},
        }
    )

    namespace = config.to_sglang()

    assert namespace.model_path == "Qwen/Qwen3-0.6B"
    assert namespace.served_model_name == "Qwen/Qwen3-0.6B"
    assert namespace.host == "0.0.0.0"
    assert namespace.port == 9000
    assert namespace.context_length == 4096
    assert namespace.tensor_parallel_size == 2
    assert namespace.data_parallel_size == 1
    assert namespace.disable_cuda_graph is True
    assert namespace.disable_radix_cache is True
    assert namespace.enable_fp32_lm_head is True
    assert namespace.mem_fraction_static == 0.8
    assert namespace.attention_backend == "triton"
    assert not hasattr(namespace, "tool_call_parser")


def test_rl_config_auto_selects_openai_client_for_sglang():
    config = RLConfig.model_validate({"trainer": {}, "orchestrator": {}, "inference": {"backend": "sglang"}})

    assert config.orchestrator.use_token_client is False
    assert config.orchestrator.client.admin_backend == "sglang"


def test_inference_config_translates_dynamo_args():
    config = InferenceConfig.model_validate(
        {
            "backend": "dynamo",
            "server": {"host": "0.0.0.0", "port": 9000},
            "model": {"name": "Qwen/Qwen3-0.6B", "max_model_len": 4096, "enforce_eager": True},
            "parallel": {"tp": 2, "dp": 1},
            "gpu_memory_utilization": 0.8,
            "dynamo": {"system_port": 9001, "discovery_backend": "file", "worker_extra": {"block_size": 64}},
        }
    )

    frontend = config.to_dynamo_frontend()
    worker = config.to_dynamo_vllm()

    assert frontend.http_host == "0.0.0.0"
    assert frontend.http_port == 9000
    assert frontend.namespace == "dynamo"
    assert frontend.discovery_backend == "file"
    assert frontend.model_name == "Qwen/Qwen3-0.6B"
    assert frontend.dyn_chat_processor == "vllm"
    assert worker.model == "Qwen/Qwen3-0.6B"
    assert worker.served_model_name == "Qwen/Qwen3-0.6B"
    assert worker.max_model_len == 4096
    assert worker.tensor_parallel_size == 2
    assert worker.enforce_eager is True
    assert worker.gpu_memory_utilization == 0.8
    assert worker.use_vllm_tokenizer is False
    assert worker.block_size == 64


def test_rl_config_auto_selects_openai_client_for_dynamo():
    config = RLConfig.model_validate({"trainer": {}, "orchestrator": {}, "inference": {"backend": "dynamo"}})

    assert config.orchestrator.use_token_client is False
    assert config.orchestrator.client.admin_backend == "dynamo"
    assert config.orchestrator.client.admin_base_url == ["http://localhost:8081"]
    assert config.orchestrator.client.skip_model_check is True
    assert all("return_token_ids" not in env.sampling.extra_body for env in config.orchestrator.train.env)


def test_rl_config_selects_sglang_nccl_trainer_backend():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"backend": "sglang"},
            "weight_broadcast": {"type": "nccl"},
        }
    )

    assert config.trainer.weight_broadcast.type == "nccl"
    assert config.trainer.weight_broadcast.target_backend == "sglang"
    assert config.orchestrator.client.admin_backend == "sglang"


def test_rl_config_rejects_sglang_nccl_dp():
    with pytest.raises(ValidationError, match="requires inference.parallel.dp = 1"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {},
                "deployment": {"type": "single_node", "num_train_gpus": 1, "num_infer_gpus": 2},
                "inference": {"backend": "sglang", "parallel": {"tp": 1}},
                "weight_broadcast": {"type": "nccl"},
            }
        )


def test_rl_config_rejects_sglang_quantized_nccl():
    with pytest.raises(ValidationError, match="does not support quantize_in_weight_transfer"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"impl": "custom"}},
                "orchestrator": {},
                "inference": {"backend": "sglang"},
                "weight_broadcast": {"type": "nccl", "quantize_in_weight_transfer": True},
            }
        )


def test_rl_config_rejects_sglang_with_explicit_token_client():
    with pytest.raises(ValidationError, match="does not support orchestrator.use_token_client"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {"use_token_client": True},
                "inference": {"backend": "sglang"},
            }
        )


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})
