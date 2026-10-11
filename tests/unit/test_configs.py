import os
from pathlib import Path
from typing import Annotated, Literal, get_args

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError
from renderers import custom_renderer_config

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import (
    EvalSourcesConfig,
    OrchestratorConfig,
    RLOnlineEvalConfig,
    ScheduledEvalConfig,
    TrainConfig,
)
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli, dump_resolved_config

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
    EnvServerConfig,
    EvalConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`."""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


def can_parse(config_cls: type, args: list[str]) -> bool:
    """Whether `cli` parses the given `@` config file args into `config_cls`."""
    try:
        cli(config_cls, args=args)
        return True
    except (ValidationError, ConfigFileError, SystemExit):
        return False


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class.

    A file that no class parses standalone is an overlay — a checked-in config that only carries
    the deltas over a shared base (e.g. the glm-4.5-air budget variants over
    `swe-budget.toml`). Retry those composed with each sibling TOML as the base: the
    documented `@ base.toml @ overlay.toml` left-to-right merge (docs/configuration.md,
    "TOML Composition").
    """
    could_parse = [can_parse(config_cls, ["@", config_file.as_posix()]) for config_cls in CONFIG_CLASSES]
    if not any(could_parse):
        sibling_bases = sorted(p for p in config_file.parent.glob("*.toml") if p != config_file)
        could_parse = [
            can_parse(config_cls, ["@", base.as_posix(), "@", config_file.as_posix()])
            for base in sibling_bases
            for config_cls in CONFIG_CLASSES
        ]
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


@pytest.mark.parametrize("loss_type", ["score_centering", "ipo", "ipo_tis"])
def test_rl_launcher_config_roundtrip(tmp_path, loss_type):
    from prime_rl.entrypoints.rl import write_config

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/basic/reverse-text/rl.toml",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run.name",
            "roundtrip",
            "--trainer.loss.type",
            loss_type,
            *(["--trainer.loss.score-centering"] if loss_type != "score_centering" else []),
            "--trainer.loss.topk" if loss_type == "score_centering" else "--trainer.loss.score-centering-topk",
            "4",
        ],
    )
    write_config(config, tmp_path, exclude={"slurm", "dry_run", "clean"})
    reloaded = cli(RLConfig, args=["@", str(tmp_path / "rl.json")])
    assert reloaded.run_dir == config.run_dir
    assert reloaded.trainer.weight_broadcast == config.trainer.weight_broadcast
    assert reloaded.orchestrator.train.source[0].sampling.logprobs == 4
    assert reloaded.inference.vllm.max_logprobs >= 6
    assert reloaded.orchestrator.train.source[0].sampling.top_k is None


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


def test_icepop_is_an_optional_loss_with_validated_ratio_bounds():
    default_config = TrainerConfig()
    assert default_config.loss.type == "ipo"

    config = TrainerConfig.model_validate({"loss": {"type": "icepop", "ratio_low": 0.2, "ratio_high": 5.0}})
    assert config.loss.type == "icepop"
    assert config.loss.ratio_low == 0.2
    assert config.loss.ratio_high == 5.0

    with pytest.raises(ValidationError, match="ratio_low must not exceed ratio_high"):
        TrainerConfig.model_validate({"loss": {"type": "icepop", "ratio_low": 5.0, "ratio_high": 0.2}})

    assert TrainerConfig.model_validate({"loss": {"type": "ppo"}}).loss.type == "ppo"
    assert TrainerConfig.model_validate({"loss": {"type": "cispo"}}).loss.type == "cispo"
    ipo_tis = TrainerConfig.model_validate({"loss": {"type": "ipo_tis"}}).loss
    assert ipo_tis.eps == 0.2
    assert ipo_tis.ratio_cap == 2.0
    assert not ipo_tis.score_centering
    assert default_config.loss.eps == 0.3
    assert default_config.loss.max_importance_ratio == 1e4
    for cap in (0.5, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            TrainerConfig.model_validate({"loss": {"type": "ipo_tis", "ratio_cap": cap}})
    with pytest.raises(ValidationError, match="score_centering_topk requires"):
        TrainerConfig.model_validate({"loss": {"type": "ipo_tis", "score_centering_topk": 128}})
    for gate in ("sample", "head"):
        with pytest.raises(ValidationError, match="score_centering_gate requires"):
            TrainerConfig.model_validate({"loss": {"type": "ipo_tis", "score_centering_gate": gate}})
        gated = {"type": "ipo_tis", "score_centering": True, "score_centering_gate": gate}
        assert TrainerConfig.model_validate({"loss": gated}).loss.score_centering_gate == gate
        for minimum in (0.0, -0.1, 1.1, float("nan")):
            with pytest.raises(ValidationError):
                TrainerConfig.model_validate({"loss": {**gated, "score_centering_min_acceptance": minimum}})
    with pytest.raises(ValidationError, match="requires eps > 0"):
        TrainerConfig.model_validate(
            {
                "loss": {
                    "type": "ipo_tis",
                    "score_centering": True,
                    "score_centering_gate": "sample",
                    "score_centering_topk": 128,
                    "eps": 0.0,
                }
            }
        )

    with pytest.raises(ValidationError, match="max_importance_ratio must be at least ratio_high"):
        TrainerConfig.model_validate({"loss": {"type": "ppo", "max_importance_ratio": 1.0, "ratio_high": 1.2}})


def test_moe_runtime_defaults_are_independent_from_dense_quantization():
    config = TrainerModelConfig.model_validate({"quantization": {"type": "mxfp8"}})

    assert config.quantization is not None and config.quantization.type == "mxfp8"
    assert config.moe.compute.type == "bf16"
    assert config.moe.compute.backend == "torch"
    assert config.moe.dispatch.type == "torch"
    assert config.moe.dispatch.transport == "bf16"


@pytest.mark.parametrize(
    "model",
    [
        {"ep": 2, "moe": {"compute": {"type": "bf16", "backend": "sonicmoe"}}},
        {
            "ep": 2,
            "moe": {"compute": {"type": "bf16", "backend": "sonicmoe"}, "dispatch": {"type": "deepep"}},
        },
        {"moe": {"compute": {"type": "deepgemm_fp8"}}},
        {"moe": {"compute": {"type": "mxfp8", "recipe": "mxfp8_rceil_wgrad_with_hp"}}},
        {"ep": 2, "moe": {"dispatch": {"type": "deepep", "num_sms": 16, "token_chunk_size": 1024}}},
        {
            "ep": 2,
            "moe": {
                "compute": {"type": "mxfp8"},
                "dispatch": {"type": "torch", "transport": "mxfp8"},
            },
        },
    ],
)
def test_supported_moe_runtime_configs(model):
    config = TrainerModelConfig.model_validate(model)
    assert config.moe.compute.resolve_layers(43) == set(range(43))


@pytest.mark.parametrize("backend", ["bf16", "deepgemm_fp8", "mxfp8"])
@pytest.mark.parametrize(
    ("selection", "num_layers", "selected"),
    [
        ("all", 10, set(range(10))),
        ("85%", 48, set(range(40))),
        ("100%", 3, {0, 1, 2}),
        ("0%", 3, set()),
        ("33.3%", 10, {0, 1, 2}),
        ([], 10, set()),
        ([2, 0], 10, {0, 2}),
        ([0, 2, 2], 3, {0, 2}),
    ],
)
def test_moe_compute_apply_to(backend, selection, num_layers, selected):
    config = TrainerModelConfig.model_validate({"moe": {"compute": {"type": backend, "apply_to": selection}}})
    config = TrainerModelConfig.model_validate_json(config.model_dump_json())
    assert config.moe.compute.resolve_layers(num_layers) == selected


@pytest.mark.parametrize("selection", ["*", "model.layers.*", "", "101%", "-1%", "nan%", [-1], [1.5], [True]])
def test_moe_compute_rejects_invalid_apply_to(selection):
    with pytest.raises(ValidationError, match="apply_to"):
        TrainerModelConfig.model_validate({"moe": {"compute": {"type": "mxfp8", "apply_to": selection}}})


def test_moe_compute_rejects_out_of_range_layer():
    config = TrainerModelConfig.model_validate({"moe": {"compute": {"type": "mxfp8", "apply_to": [3]}}})
    with pytest.raises(ValueError, match="out of range"):
        config.moe.compute.resolve_layers(3)


@pytest.mark.parametrize(("selection", "selected"), [("all", {0, 1, 2}), ("50%", {0}), ("[0, 2]", {0, 2})])
def test_moe_compute_apply_to_cli(selection, selected):
    config = cli(TrainerModelConfig, args=["--moe.compute.type", "mxfp8", "--moe.compute.apply-to", selection])
    assert config.moe.compute.resolve_layers(3) == selected


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            {
                "ep": 2,
                "moe": {
                    "compute": {"type": "bf16"},
                    "dispatch": {"type": "torch", "transport": "mxfp8"},
                },
            },
            "MXFP8 transport requires",
        ),
        (
            {
                "ep": 2,
                "moe": {
                    "compute": {"type": "mxfp8"},
                    "dispatch": {"type": "deepep"},
                },
            },
            "does not support DeepEP",
        ),
    ],
)
def test_invalid_moe_runtime_configs_fail(model, message):
    with pytest.raises(ValidationError, match=message):
        TrainerModelConfig.model_validate(model)


@pytest.mark.parametrize(
    "removed",
    [
        {"ep_comm_backend": "torch"},
        {"deepep_num_sms": 20},
        {"deepep_token_chunk_size": 1024},
        {"quantization": {"type": "fp8", "enable_grouped_gemm": True}},
        {"quantization": {"type": "mxfp8", "enable_a2a": True}},
    ],
)
def test_removed_moe_runtime_fields_are_rejected(removed):
    with pytest.raises(ValidationError):
        TrainerModelConfig.model_validate(removed)


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_optimizer_state_offload_keeps_legacy_default(config_cls):
    config = config_cls.model_validate({})

    assert config.model.optim_cpu_offload is True
    assert config.model.full_offload is None


def test_moe_router_dtype_auto_resolves_per_trainer():
    """``moe_router_dtype='auto'`` (the default) resolves to fp32 for RL and bf16 for SFT; explicit values are kept."""
    assert TrainerConfig.model_validate({}).model.moe_router_dtype == "float32"
    assert SFTConfig.model_validate({}).model.moe_router_dtype == "bfloat16"

    for config_cls in (TrainerConfig, SFTConfig):
        for dtype in ("bfloat16", "float32"):
            assert config_cls.model_validate({"model": {"moe_router_dtype": dtype}}).model.moe_router_dtype == dtype


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_disables_gradient_clipping(config_cls):
    with pytest.warns(UserWarning, match="Gradient clipping prevents optimizer-in-backward"):
        config = config_cls.model_validate(
            {
                "model": {"optim_cpu_offload": False, "full_offload": True},
                "optim": {"max_norm": 1.0},
            }
        )

    assert config.optim.max_norm is None


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_requires_supported_optimizer(config_cls):
    with pytest.raises(ValidationError, match="Full optimizer offload only supports AdamW and SignSGD"):
        config_cls.model_validate(
            {
                "model": {"optim_cpu_offload": False, "full_offload": True},
                "optim": {"type": "muon", "max_norm": None},
            }
        )


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_accepts_sign_sgd(config_cls):
    config = config_cls.model_validate(
        {
            "model": {"optim_cpu_offload": False, "full_offload": True},
            "optim": {"type": "sign_sgd", "max_norm": None},
        }
    )
    assert config.model.full_offload is not None
    assert config.optim.type == "sign_sgd"


def test_resolved_json_roundtrips_explicit_none(tmp_path):
    """An explicit None override survives the write/re-parse round-trip used by launches:
    resolved configs are JSON, which keeps nulls (TOML cannot)."""
    import json

    config = cli(TrainerConfig, args=["--model.compile", "None", "--optim.max_norm", "None"])
    assert config.model.compile is None
    assert config.optim.max_norm is None

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(dump_resolved_config(config)))
    reloaded = cli(TrainerConfig, args=["@", str(path)])
    assert reloaded.model.compile is None
    assert reloaded.optim.max_norm is None
    assert reloaded == config


def test_env_algo_inherits_the_group_algo():
    config = OrchestratorConfig.model_validate(
        {
            "renderer": {"name": "qwen3"},  # echo needs the renderer's role attribution
            "train": {
                "algo": {"type": "echo", "roles": {"user": {"alpha": 0.1}}},
                "source": [
                    {"env": {"taskset": {"id": "reverse-text"}}, "algo": {"type": "grpo"}},
                    {"env": {"taskset": {"id": "reverse-text"}}, "name": "b"},
                    {"env": {"taskset": {"id": "reverse-text"}}, "name": "c", "algo": {"roles": {"tool": {}}}},
                ],
            },
        }
    )
    env_a, env_b, env_c = config.train.source
    # A different type is the env's own algorithm; no type keeps the group's.
    assert env_a.algo.type == "grpo"
    assert env_b.algo.type == "echo" and env_b.algo.roles.user.alpha == 0.1
    assert env_c.algo.type == "echo" and env_c.algo.roles.user.alpha == 0.1 and env_c.algo.roles.tool is not None

    # Resolved configs round-trip.
    dumped = config.model_dump(exclude_none=True)
    reloaded = OrchestratorConfig.model_validate(dumped)
    assert [env.algo.type for env in reloaded.train.source] == ["grpo", "echo", "echo"]


def test_sources_inherit_the_group_fields_they_leave_unset():
    config = EvalConfig.model_validate(
        {
            "r": 4,
            "sampling": {"temperature": 0.5, "extra_body": {"a": 1}},
            "select": {"limit": 8, "include": {"idx": [":100"]}},
            "env": {"retries": {"max_retries": 3}},
            "source": [
                {"env": {"taskset": {"id": "gsm8k"}}},
                {
                    "name": "b",
                    "env": {"taskset": {"id": "gsm8k"}},
                    "group_size": 2,
                    "sampling": {"extra_body": {"b": 2}},
                    "select": {"include": {"names": ["x"]}},
                },
            ],
        }
    )
    a, b = config.source
    assert (a.group_size, b.group_size) == (4, 2)
    assert b.sampling.temperature == 0.5 and b.sampling.extra_body == {"a": 1, "b": 2}
    assert b.select.limit == 8 and b.select.include.idx == [":100"] and b.select.include.names == ["x"]
    assert a.env.retries.max_retries == 3 and b.env.retries.max_retries == 3
    reloaded = EvalConfig.model_validate(config.model_dump(mode="json"))
    assert reloaded.source[1].model_dump() == b.model_dump()

    online = RLOnlineEvalConfig.model_validate(
        {
            "interval": 5,
            "source": [
                {"env": {"taskset": {"id": "gsm8k"}}},
                {"name": "b", "env": {"taskset": {"id": "gsm8k"}}, "interval": 2},
            ],
        }
    )
    assert online.intervals == {"gsm8k": 5, "b": 2}


def test_train_sources_can_use_their_own_group_sizes():
    def orchestrator(batch_size: int, **concurrency) -> OrchestratorConfig:
        return OrchestratorConfig.model_validate(
            {
                "batch_size": batch_size,
                "concurrency": concurrency,
                "train": {
                    "group_size": 16,
                    "source": [
                        {"env": {"taskset": {"id": "reverse-text"}}},
                        {"env": {"taskset": {"id": "reverse-text"}}, "name": "b", "group_size": 8},
                    ],
                },
            }
        )

    assert [source.group_size for source in orchestrator(64).train.source] == [16, 8]
    with pytest.raises(ValidationError, match="divisible by every train source"):
        orchestrator(40)
    with pytest.raises(ValidationError, match="largest train group_size"):
        orchestrator(64, max_inflight=8)


@pytest.mark.parametrize(
    "group", [TrainConfig, EvalSourcesConfig, ScheduledEvalConfig, EvalConfig, SFTOnlineEvalConfig]
)
def test_group_defaults_match_source_defaults(group):
    # An unset group field is not layered, so its default must be the source's.
    def default(field):
        value = field.get_default(call_default_factory=True)
        return value.model_dump() if isinstance(value, BaseModel) else value

    (source,) = get_args(group.model_fields["source"].annotation)
    for name, field in group.model_fields.items():
        if name not in ("source", "env") and name in source.model_fields:
            assert default(field) == default(source.model_fields[name]), f"{group.__name__}.{name}"


def test_policy_sources_accept_different_top_p_values():
    with pytest.warns(UserWarning, match="defaulting top_k"):
        config = OrchestratorConfig.model_validate(
            {
                "renderer": {"name": "qwen3"},
                "train": {
                    "source": [
                        {
                            "name": "top-p-95",
                            "env": {"taskset": {"id": "reverse-text"}},
                            "sampling": {"top_p": 0.95},
                        },
                        {
                            "name": "top-p-97",
                            "env": {"taskset": {"id": "reverse-text"}},
                            "sampling": {"top_p": 0.97},
                        },
                    ]
                },
            }
        )

    assert [source.sampling.top_k for source in config.train.source] == [512, 512]


def test_policy_sources_reject_mixed_top_k_capture():
    with pytest.warns(UserWarning, match="defaulting top_k"):
        with pytest.raises(ValidationError, match="cannot mix top_k > 0 and top_k = -1"):
            OrchestratorConfig.model_validate(
                {
                    "renderer": {"name": "qwen3"},
                    "train": {
                        "source": [
                            {
                                "name": "truncated",
                                "env": {"taskset": {"id": "reverse-text"}},
                                "sampling": {"top_p": 0.95},
                            },
                            {
                                "name": "untruncated",
                                "env": {"taskset": {"id": "reverse-text"}},
                            },
                        ]
                    },
                }
            )


def test_single_node_auto_inference_ports_follow_server_port():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"server": {"port": 8001}, "vllm": {"tensor_parallel_size": 1}},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 4,
                "num_train_gpus": 2,
                "num_infer_gpus": 2,
            },
        }
    )

    assert config.inference is not None
    assert config.inference.vllm.data_parallel_size == 2
    assert config.inference.backend_port == 8101
    assert config.orchestrator.model.client.admin_base_url == ["http://localhost:8101/v1"]


def test_multi_node_auto_inference_parallelism():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"vllm": {"tensor_parallel_size": 4}},
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 8,
                "num_train_nodes": 1,
                "num_infer_nodes": 2,
            },
            "slurm": {},
        }
    )

    assert config.inference is not None
    assert config.inference.vllm.data_parallel_size_local == 2
    assert config.inference.vllm.data_parallel_size == 2


def test_orchestrator_vlm_requires_renderer():
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {
                    "name": "Qwen/Qwen3-VL-4B-Instruct",
                    "vlm": {
                        "vision_encoder_attr": "model.visual",
                        "language_model_attr": "model.language_model",
                    },
                },
                "renderer": None,
            }
        )

    config = OrchestratorConfig.model_validate(
        {
            "model": {
                "name": "Qwen/Qwen3-VL-4B-Instruct",
                "vlm": {
                    "vision_encoder_attr": "model.visual",
                    "language_model_attr": "model.language_model",
                },
            },
        }
    )

    assert config.renderer is not None


def test_trainer_rejects_vlm_cp_with_ring():
    config = {
        "model": {
            "cp": 2,
            "optimization_dtype": "bfloat16",
            "reduce_dtype": "bfloat16",
            "vlm": {
                "vision_encoder_attr": "model.visual",
                "language_model_attr": "model.language_model",
            },
        },
    }

    with pytest.raises(ValidationError, match="cp_style='ulysses'"):
        TrainerConfig.model_validate(config)


def test_shared_model_name_propagates_to_subconfigs():
    model_name = "PrimeIntellect/test-model"
    config = RLConfig.model_validate(
        {
            "model": {"name": model_name},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.trainer.model.name == model_name
    assert config.orchestrator.model.name == model_name
    assert config.inference is not None and config.inference.vllm.model == model_name
    assert config.trainer.tokenizer.name == model_name
    assert config.orchestrator.tokenizer.name == model_name


def test_shared_tokenizer_propagates_when_subconfigs_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"name": "my-tokenizer"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-tokenizer"
    assert config.orchestrator.tokenizer.name == "my-tokenizer"


def test_shared_and_sub_tokenizer_name_conflict_raises():
    """Setting tokenizer.name in both [tokenizer] and [trainer.tokenizer]
    is a config conflict — the sub-config would silently win, and any later
    CLI override of [tokenizer].name would silently no-op for the trainer."""
    with pytest.raises(ValidationError, match=r"tokenizer.name.*trainer.tokenizer.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "my-model"},
                "tokenizer": {"name": "shared-tok"},
                "trainer": {"tokenizer": {"name": "trainer-tok"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_tokenizer_name_falls_back_to_model_name_when_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"trust_remote_code": True},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-model"
    assert config.orchestrator.tokenizer.name == "my-model"
    assert config.trainer.tokenizer.trust_remote_code is True
    assert config.orchestrator.tokenizer.trust_remote_code is True


def test_explicit_subconfig_tokenizer_name_survives_shared_model_propagation():
    """Regression: shared ``[model] name = "M"`` must propagate model names but
    must NOT clobber an explicit ``[orchestrator.tokenizer] name = "T"``.

    This is the case that the old RL-level ``auto_setup_tokenizer`` fix-up got
    wrong: it unconditionally re-derived ``orchestrator.tokenizer.name`` from
    ``orchestrator.model.name`` after propagation, silently overriding
    the user's explicit value. The ``mode="before"`` ``auto_setup_shared_configs``
    propagator fixes this because it propagates the model name into the raw
    dict before sub-configs are built, so ``OrchestratorConfig``'s own
    ``auto_setup_tokenizer`` (mode=after) sees the resolved name *and* the
    explicit user-set tokenizer name, and the ``fill``-if-absent semantic
    leaves the explicit value alone.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "M"},
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "default"},
                "tokenizer": {"name": "explicit-orch-tok"},
            },
        }
    )
    # Shared model.name reached every sub-config that didn't override it.
    assert config.trainer.model.name == "M"
    assert config.orchestrator.model.name == "M"
    # Trainer didn't specify a tokenizer, so it falls back to the propagated model name.
    assert config.trainer.tokenizer.name == "M"
    # Orchestrator's explicit tokenizer name survived.
    assert config.orchestrator.tokenizer.name == "explicit-orch-tok"


def test_tokenizer_chat_template_mismatch_raises():
    with pytest.raises(ValidationError, match="chat_template"):
        RLConfig.model_validate(
            {
                "trainer": {"tokenizer": {"chat_template": "A"}},
                "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "B"}},
            }
        )


def test_shared_seq_len_propagates_to_subconfigs():
    config = RLConfig.model_validate(
        {
            "seq_len": 4096,
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.seq_len == 4096
    assert config.orchestrator.seq_len == 4096


def test_shared_and_sub_seq_len_conflict_raises():
    """Setting seq_len at the shared level and on a sub-config is a conflict —
    forces the user to pick one place to express the value rather than
    relying on the silent 'sub wins' rule."""
    with pytest.raises(ValidationError, match=r"seq_len.*trainer.model.seq_len"):
        RLConfig.model_validate(
            {
                "seq_len": 4096,
                "trainer": {"model": {"seq_len": 8192}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_model_name_conflict_raises():
    """Setting model.name at the shared level and on a sub-config is a conflict."""
    with pytest.raises(ValidationError, match=r"model.name.*trainer.model.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "X"},
                "trainer": {"model": {"name": "Y"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_max_steps_conflict_raises():
    """Top-level scalar shared fields also participate in the mutex check."""
    with pytest.raises(ValidationError, match=r"max_steps.*orchestrator.max_steps"):
        RLConfig.model_validate(
            {
                "max_steps": 100,
                "trainer": {},
                "orchestrator": {"renderer": {"name": "default"}, "max_steps": 200},
            }
        )


def test_trainer_chat_template_cascades_to_inference():
    """``[trainer.tokenizer] chat_template`` set directly (no shared
    ``[tokenizer] chat_template``) must still reach
    ``inference.vllm.chat_template`` so vLLM's ``--chat-template`` is wired
    up. Regression: the original ``auto_setup_tokenizer`` cascaded this; the
    refactored propagator must keep doing it."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"tokenizer": {"chat_template": "TPL"}},
            "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "TPL"}},
            "inference": {},
        }
    )
    assert config.trainer.tokenizer.chat_template == "TPL"
    assert config.orchestrator.tokenizer.chat_template == "TPL"
    assert config.inference is not None
    assert config.inference.vllm.chat_template == "TPL"


def test_shared_wandb_fields_propagate_to_subconfigs():
    """Every ``SharedWandbConfig`` leaf (project, entity, name, group, tags,
    offline) propagates to both trainer.monitors.wandb and
    orchestrator.monitors.wandb. Regression for a miss in the inline propagator."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "monitors": {
                "wandb": {
                    "project": "shared-proj",
                    "entity": "shared-entity",
                    "name": "shared-name",
                    "group": "shared-group",
                    "tags": ["a", "b"],
                    "offline": False,
                }
            },
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    for component in (config.trainer.monitors.wandb, config.orchestrator.monitors.wandb):
        assert component is not None
        assert component.project == "shared-proj"
        assert component.entity == "shared-entity"
        assert component.name == "shared-name"
        assert component.group == "shared-group"
        assert component.tags == ["a", "b"]
        assert component.offline is False


def test_shared_monitor_disable_and_prime_propagate():
    """CLI ``--no-monitors.wandb`` / ``--no-monitors.file`` (which land as the string
    "None") propagate the disable to both sub-configs, whose monitors default to
    enabled; a shared ``[monitors.prime]`` reaches the orchestrator only."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "monitors": {"wandb": "None", "file": "None", "prime": {"name": "shared-prime"}},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.monitors.wandb is None and config.trainer.monitors.file is None
    assert config.orchestrator.monitors.wandb is None and config.orchestrator.monitors.file is None
    assert config.orchestrator.monitors.prime is not None
    assert config.orchestrator.monitors.prime.name == "shared-prime"


def test_empty_shared_ckpt_block_does_not_conflict_with_subconfig_ckpt():
    """An empty shared [ckpt] block is a presence-only signal, not a field
    setting — it should not conflict with a non-empty [trainer.ckpt]."""
    config = RLConfig.model_validate(
        {
            "ckpt": {},  # empty block, no field set
            "trainer": {"ckpt": {"interval": 50}},
            "orchestrator": {"renderer": {"name": "default"}, "ckpt": {"interval": 50}},
        }
    )
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.interval == 50


def test_shared_and_subconfig_disjoint_fields_coexist():
    """Per-field mutex only forbids conflicts on the SAME field — disjoint
    fields in [model] vs [trainer.model] are fine."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"model": {"attn": "flash_attention_3"}},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.name == "Qwen/Qwen3-0.6B"
    assert config.trainer.model.attn == "flash_attention_3"


def test_run_dir_propagates_through_cli(tmp_path):
    """Sub-configs receive the run directory (output_dir / run.name) resolved from the CLI."""
    toml_path = tmp_path / "cfg.toml"
    write_toml(
        toml_path,
        {
            "max_steps": 1,
            "seq_len": 128,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "monitors": {"wandb": {}},
            "trainer": {},
            "orchestrator": {"batch_size": 16, "train": {"group_size": 1}},
            "inference": {},
        },
    )
    shared_out = tmp_path / "shared"
    config = cli(RLConfig, args=["@", str(toml_path), "--output-dir", str(shared_out), "--run.name", "my-exp"])
    assert config.run_dir == shared_out / "my-exp"
    assert config.trainer.output_dir == shared_out / "my-exp"
    assert config.orchestrator.output_dir == shared_out / "my-exp"
    # Unset monitor names inherit run.name
    assert config.monitors.wandb is not None and config.monitors.wandb.name == "my-exp"
    assert config.trainer.monitors.wandb is not None and config.trainer.monitors.wandb.name == "my-exp"
    assert config.orchestrator.monitors.wandb is not None and config.orchestrator.monitors.wandb.name == "my-exp"


def test_orchestrator_renderer_auto_rejects_unmapped_model():
    """Default ``renderer`` (AutoRendererConfig) must reject models not in MODEL_RENDERER_MAP."""
    with pytest.raises(ValidationError, match="silently fall back to DefaultRenderer"):
        OrchestratorConfig.model_validate({"model": {"name": "not-a-real-org/not-a-real-model"}})


def test_orchestrator_renderer_auto_accepts_mapped_model():
    """The default Qwen model is in MODEL_RENDERER_MAP and should validate cleanly."""
    config = OrchestratorConfig.model_validate({"model": {"name": "Qwen/Qwen3-0.6B"}})
    assert config.renderer is not None
    assert config.renderer.name == "auto"


def test_sft_renderer_auto_accepts_prime_qwen_model():
    config = SFTConfig.model_validate({"model": {"name": "PrimeIntellect/Qwen3-0.6B"}})
    assert config.renderer.name == "auto"


def test_sft_rejects_default_renderer_for_real_data():
    with pytest.raises(ValidationError, match="requires a typed renderer"):
        SFTConfig.model_validate({"renderer": {"name": "default"}})


def test_sft_allows_unused_default_renderer_for_fake_data():
    config = SFTConfig.model_validate(
        {
            "data": {"type": "fake"},
            "renderer": {"name": "default"},
        }
    )
    assert config.renderer.name == "default"


def test_orchestrator_explicit_renderer_skips_unmapped_check():
    """Explicit renderer.name bypasses the auto-resolution check — user opted in."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "qwen3"


def test_orchestrator_renderer_none_rejected():
    """A renderer is required (training is renderer-only): the non-optional type rejects None."""
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {"name": "not-a-real-org/not-a-real-model"},
                "renderer": None,
            }
        )


def test_orchestrator_explicit_default_renderer_with_unmapped_model():
    """renderer.name='default' is an explicit opt-in to DefaultRenderer and must pass."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "default", "tool_parser": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "default"
    assert config.renderer.tool_parser == "qwen3"


def test_shared_model_name_resolves_inference_parsers():
    """Shared [model] name must reach inference.vllm BEFORE VllmConfig's after-validator
    runs auto_resolve_parsers — i.e. the parsers resolve from the propagated name, not
    from an empty default.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.inference is not None
    assert config.inference.vllm.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config.inference.vllm.tool_call_parser == "qwen3_coder"


def test_explicit_inference_parser_wins_over_auto():
    """Explicit inference.vllm.tool_call_parser is preserved even when the shared model
    name would otherwise auto-resolve to something else."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {"vllm": {"tool_call_parser": "hermes"}},
        }
    )
    assert config.inference is not None
    assert config.inference.vllm.tool_call_parser == "hermes"


def test_combined_replay_uses_v2_runner(monkeypatch):
    from prime_rl.inference.server import setup_vllm_env

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    config = InferenceConfig(
        enable_return_sampling_mask=True,
        vllm={"enable_return_routed_experts": True},
    )

    setup_vllm_env(config)

    assert config.enable_return_sampling_mask is True
    assert config.vllm.enable_return_routed_experts is True
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


CUSTOM_RENDERER_SOURCE = """
from typing import Literal

from renderers.configs import BaseRendererConfig


class InstructedRendererConfig(BaseRendererConfig):
    name: Literal["instructed"] = "instructed"
    _template_fields = frozenset({"instruction"})

    instruction: str = "Think step by step."


class InstructedRenderer:
    config_class = InstructedRendererConfig
"""


@pytest.fixture
def custom_renderer_import_path(tmp_path):
    path = tmp_path / "instructed_renderer.py"
    path.write_text(CUSTOM_RENDERER_SOURCE)
    return f"{path}:InstructedRenderer"


def test_sft_config_accepts_custom_renderer(tmp_path, custom_renderer_import_path):
    config_file = tmp_path / "sft.toml"
    config_file.write_text(
        tomli_w.dumps(
            {
                "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
                "data": {"name": "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"},
                "renderer": {"name": "custom", "import_path": custom_renderer_import_path, "instruction": "Be brief."},
            }
        )
    )

    config = cli(SFTConfig, args=["@", str(config_file)])

    assert config.renderer.name == "custom"
    assert custom_renderer_config(config.renderer).name == "instructed"
    assert custom_renderer_config(config.renderer).instruction == "Be brief."


def test_orchestrator_config_accepts_custom_renderer(tmp_path, custom_renderer_import_path):
    config_file = tmp_path / "orch.toml"
    config_file.write_text(
        tomli_w.dumps(
            {
                "model": {"name": "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"},
                "renderer": {"name": "custom", "import_path": custom_renderer_import_path},
            }
        )
    )

    config = cli(OrchestratorConfig, args=["@", str(config_file)])

    assert config.renderer.name == "custom"
    assert custom_renderer_config(config.renderer).instruction == "Think step by step."


def test_custom_renderer_rejects_unknown_fields(tmp_path, custom_renderer_import_path):
    config_file = tmp_path / "sft.toml"
    config_file.write_text(
        tomli_w.dumps(
            {
                "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
                "renderer": {"name": "custom", "import_path": custom_renderer_import_path, "depth": 3},
            }
        )
    )

    with pytest.raises(ConfigFileError, match="depth"):
        cli(SFTConfig, args=["@", str(config_file)])
