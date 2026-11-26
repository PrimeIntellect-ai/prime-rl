import importlib
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Literal, get_args

import pytest
import tomli_w
import verifiers.v1 as vf
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError
from renderers import custom_renderer_config

from prime_rl.configs.algorithm import GRPOAlgoConfig, LinearLengthPenaltyConfig, LinearSubagentLengthPenaltyConfig
from prime_rl.configs.env_worker_pool import EnvWorkerPoolConfig
from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import (
    EvalSourcesConfig,
    OrchestratorConfig,
    RLOnlineEvalConfig,
    ScheduledEvalConfig,
    TrainConfig,
    TrainSourceConfig,
)
from prime_rl.configs.orchestrator import (
    ModelConfig as OrchestratorModelConfig,
)
from prime_rl.configs.pretrain import PretrainConfig, PretrainDataConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig, SFTDataConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import MODES, BaseConfig, check_mode, cli, dump_resolved_config

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    PretrainConfig,
    OrchestratorConfig,
    InferenceConfig,
    EnvWorkerPoolConfig,
    EvalConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/`, `examples/`, `recipes/` or `tests/regression/configs/`."""
    roots = [Path("configs"), Path("examples"), Path("recipes"), Path("tests/regression/configs")]
    return [path for root in roots for path in root.rglob("*.toml")]


ENTRYPOINT_CONFIGS = {
    "rl": RLConfig,
    "trainer": TrainerConfig,
    "sft": SFTConfig,
    "pretrain": PretrainConfig,
    "orchestrator": OrchestratorConfig,
    "inference": InferenceConfig,
    "env-worker-pool": EnvWorkerPoolConfig,
    "eval": EvalConfig,
}

# A launch command in a config's comments, e.g. `#   uv run rl @ base.toml @ overlay.toml --max-steps 5`.
# Only the `@` file stack counts; lines with placeholders instead of TOML paths don't match.
LAUNCH_COMMAND = re.compile(r"^#\s*uv run ([\w-]+)((?:\s+@\s+\S+\.toml)+)")


def get_launch_commands() -> list[tuple[str, str, list[Path]]]:
    """(location, entrypoint, config stack) for each launch command documented in a config file."""
    commands = []
    for config_file in get_config_files():
        for number, line in enumerate(config_file.read_text().splitlines(), start=1):
            if match := LAUNCH_COMMAND.match(line):
                stack = [Path(path) for path in match[2].split()[1::2]]
                commands.append((f"{config_file.as_posix()}:{number}", match[1], stack))
    return commands


LAUNCH_COMMANDS = get_launch_commands()


def can_parse(config_cls: type, args: list[str]) -> bool:
    """Whether `cli` parses the given `@` config file args into `config_cls`."""
    try:
        cli(config_cls, args=args)
        return True
    except (ValidationError, ConfigFileError, SystemExit):
        return False


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that every config file parses on its own into at least one config class, or is an overlay.

    An overlay only carries the deltas over a base (e.g. the glm-4.5-air budget variants over
    `swe-budget.toml`). It must be part of a launch command documented in a config's comments,
    which `test_launch_commands` parses.
    """
    if any(config_file in stack for _, _, stack in LAUNCH_COMMANDS):
        return
    assert any(can_parse(config_cls, ["@", config_file.as_posix()]) for config_cls in CONFIG_CLASSES), (
        f"{config_file} parses into no config class on its own. If it is an overlay, document how to "
        f"compose it in a header comment, e.g. `#   uv run rl @ base.toml @ {config_file.as_posix()}`."
    )


@pytest.mark.parametrize(
    ("entrypoint", "stack"),
    [(entrypoint, stack) for _, entrypoint, stack in LAUNCH_COMMANDS],
    ids=[location for location, _, _ in LAUNCH_COMMANDS],
)
def test_launch_commands(entrypoint: str, stack: list[Path]):
    """Tests that each launch command documented in a config file parses with its entrypoint's config class."""
    assert entrypoint in ENTRYPOINT_CONFIGS, f"Unknown entrypoint `{entrypoint}`"
    missing = [path for path in stack if not path.is_file()]
    assert not missing, f"Launch command references missing files: {missing}"
    args = [arg for path in stack for arg in ("@", path.as_posix())]
    cli(ENTRYPOINT_CONFIGS[entrypoint], args=args)


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


def test_rl_launcher_config_roundtrip(tmp_path):
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
            "icepop",
            "--trainer.loss.score-centering",
            "--trainer.loss.score-centering-topk",
            "4",
        ],
    )
    write_config(config, tmp_path, exclude={"slurm", "dry_run", "clean"})
    reloaded = cli(RLConfig, args=["@", str(tmp_path / "rl.json")])
    assert reloaded.run_dir == config.run_dir
    assert reloaded.trainer.weight_broadcast == config.trainer.weight_broadcast
    assert reloaded.orchestrator.train.source[0].sampling.logprobs == 4
    assert reloaded.inference.enable_return_sampling_mask
    assert reloaded.orchestrator.train.source[0].sampling.top_k is None


def test_pretrain_config_roundtrip(tmp_path):
    """A minimal pretrain TOML gets the pretraining defaults and survives the launcher's resolved config, which
    `uv run sft` rejects."""
    from prime_rl.entrypoints.launcher import write_config

    write_toml(tmp_path / "pretrain.toml", {"data": {"source": [{"name": "web", "path": str(tmp_path / "tokens")}]}})
    config = cli(PretrainConfig, args=["@", str(tmp_path / "pretrain.toml"), "--output-dir", str(tmp_path)])
    assert config.data.type == "datatrove"
    assert (config.optim.type, config.optim.lr, config.optim.weight_decay) == ("adamw", 1e-3, 0.01)
    assert (config.scheduler.type, config.scheduler.warmup_steps, config.scheduler.decay_steps) == ("linear", 10, 0)

    write_config(config, tmp_path / "pretrain.json")
    assert cli(PretrainConfig, args=["@", str(tmp_path / "pretrain.json")]) == config
    assert not can_parse(SFTConfig, ["@", str(tmp_path / "pretrain.json")])

    muon = cli(PretrainConfig, args=["@", str(tmp_path / "pretrain.toml"), "--optim.type", "muon"])
    assert muon.optim.lr == 1e-3
    warmup = cli(PretrainConfig, args=["@", str(tmp_path / "pretrain.toml"), "--scheduler.warmup-steps", "100"])
    assert (warmup.scheduler.warmup_steps, warmup.scheduler.decay_steps) == (100, 0)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"source": [{"name": "a", "path": "a"}, {"name": "a", "path": "b"}]}, "names must be unique"),
        (
            {"source": [{"name": "a", "path": "a", "weight": 1.0}, {"name": "b", "path": "b"}]},
            "on every source or on none",
        ),
        (
            {"source": [{"name": "a", "path": "a", "weight": 0.0}, {"name": "b", "path": "b", "weight": 0.0}]},
            "more than 0",
        ),
    ],
)
def test_pretrain_data_rejects_invalid_sources(data: dict, error: str):
    with pytest.raises(ValidationError, match=error):
        PretrainDataConfig(**data)


def test_repetitions_need_max_steps():
    sources = {"source": [{"name": "a", "path": "a", "repetitions": 2.0}]}
    with pytest.raises(ValidationError, match="needs max_steps"):
        PretrainConfig.model_validate({"data": sources})


def test_val_set_defaults(tmp_path):
    """[[val.set]] are scored at the training seq_len unless they set their own, in rows of val.micro_batch_tokens:
    the training micro-batch or the longest set seq_len, whichever is larger, rounded up to a multiple of the CP
    degree."""
    path = str(tmp_path / "tokens")
    val_sets = [
        {"type": "documents", "name": "short", "path": path},
        {"type": "documents", "name": "long", "path": path, "seq_len": 4096},
    ]
    write_toml(
        tmp_path / "pretrain.toml",
        {
            "data": {"source": [{"name": "web", "path": path}], "seq_len": 1024, "micro_batch_size": 2},
            "val": {"set": val_sets},
        },
    )
    args = ["@", str(tmp_path / "pretrain.toml"), "--output-dir", str(tmp_path)]
    config = cli(PretrainConfig, args=args)
    assert [val_set.seq_len for val_set in config.val.set] == [1024, 4096]
    assert config.val.micro_batch_tokens == 4096
    assert not can_parse(PretrainConfig, [*args, "--val.micro-batch-tokens", "2048"])
    odd_set = json.dumps([{**val_sets[1], "seq_len": 4097}])
    cp_args = [*args, "--model.cp", "2", "--data.micro-batch-size", "1", "--val.set", odd_set]
    assert cli(PretrainConfig, args=cp_args).val.micro_batch_tokens == 4098


@pytest.mark.parametrize("mode", MODES)
def test_launch_runs_the_entrypoint_of_the_mode(tmp_path, monkeypatch, mode: str):
    """`uv run launch` reads `mode` from the config files and runs that entrypoint with the same arguments."""
    from prime_rl.entrypoints import launch

    calls = []
    monkeypatch.setattr(importlib.import_module(f"prime_rl.entrypoints.{mode}"), "main", lambda: calls.append(mode))
    write_toml(tmp_path / "config.toml", {"mode": mode})
    monkeypatch.setattr(sys, "argv", ["launch", "@", str(tmp_path / "config.toml")])
    launch.main()
    assert calls == [mode]


def test_launch_needs_a_mode(tmp_path, monkeypatch):
    from prime_rl.entrypoints import launch

    write_toml(tmp_path / "config.toml", {"max_steps": 1})
    monkeypatch.setattr(sys, "argv", ["launch", "@", str(tmp_path / "config.toml")])
    with pytest.raises(SystemExit):
        launch.main()


def test_mode_must_match_the_entrypoint(tmp_path):
    """A mode entrypoint runs only its own mode, and an internal entrypoint none, from a TOML or a resolved JSON
    config."""
    write_toml(tmp_path / "pretrain.toml", {"mode": "pretrain"})
    (tmp_path / "pretrain.json").write_text(json.dumps({"mode": "pretrain"}))
    for file in ("pretrain.toml", "pretrain.json"):
        args = ["@", str(tmp_path / file)]
        check_mode("pretrain", args)
        for entrypoint in ("sft", "trainer"):
            with pytest.raises(SystemExit):
                check_mode(entrypoint, args)


def test_config_files_with_different_modes_conflict(tmp_path):
    write_toml(tmp_path / "base.toml", {"mode": "rl"})
    write_toml(tmp_path / "override.toml", {"mode": "sft"})
    with pytest.raises(SystemExit, match="conflicting modes"):
        check_mode("rl", ["@", str(tmp_path / "base.toml"), "@", str(tmp_path / "override.toml")])


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


@pytest.mark.parametrize(
    "penalty,expected_type",
    [
        ({}, LinearLengthPenaltyConfig),
        ({"num_output_tokens_weight": 0.5}, LinearLengthPenaltyConfig),
        ({"type": "linear"}, LinearLengthPenaltyConfig),
        ({"type": "linear-subagent"}, LinearSubagentLengthPenaltyConfig),
        ({"type": "linear-subagent", "subagent_token_weight": 0.2}, LinearSubagentLengthPenaltyConfig),
    ],
)
def test_length_penalty_variants_are_opt_in(tmp_path, penalty, expected_type):
    assert GRPOAlgoConfig().length_penalty is None
    direct = GRPOAlgoConfig.model_validate({"length_penalty": penalty})
    write_toml(tmp_path / "penalty.toml", {"length_penalty": penalty})
    from_toml = cli(GRPOAlgoConfig, args=["@", str(tmp_path / "penalty.toml")])
    assert type(direct.length_penalty) is expected_type
    assert from_toml.length_penalty == direct.length_penalty
    if expected_type is LinearSubagentLengthPenaltyConfig:
        assert direct.length_penalty.subagent_token_weight == penalty.get("subagent_token_weight", 0.0)
    else:
        assert "subagent_token_weight" not in direct.length_penalty.model_dump()
    with pytest.raises(ValidationError):
        GRPOAlgoConfig.model_validate({"length_penalty": {"type": "linear", "subagent_token_weight": 0.0}})


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
    assert config.loss.score_centering_topk == 128
    assert not config.loss.score_centering
    for unsupported in (
        {"type": "ipo_tis"},
        {"type": "score_centering"},
        {"type": "ipo", "score_centering": True},
        {"type": "icepop", "score_centering_topk": None},
    ):
        with pytest.raises(ValidationError):
            TrainerConfig.model_validate({"loss": unsupported})
    assert config.loss.ratio_low == 0.2
    assert config.loss.ratio_high == 5.0

    with pytest.raises(ValidationError, match="ratio_low must not exceed ratio_high"):
        TrainerConfig.model_validate({"loss": {"type": "icepop", "ratio_low": 5.0, "ratio_high": 0.2}})


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
        {"ep": 2, "moe": {"dispatch": {"type": "deepep", "num_sms": 16}}},
        {
            "ep": 2,
            "moe": {
                "compute": {"type": "mxfp8"},
                "dispatch": {"type": "torch", "transport": "mxfp8"},
            },
        },
        {
            "ep": 2,
            "moe": {
                "compute": {"type": "mxfp8", "recipe": "mxfp8_rceil_wgrad_with_hp"},
                "dispatch": {"type": "deepep"},
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
            {"ep": 8, "moe": {"dispatch": {"type": "mega_moe", "precision": "fp8", "llep": {}}}},
            "llep requires",
        ),
    ],
)
def test_invalid_moe_runtime_configs_fail(model, message):
    with pytest.raises(ValidationError, match=message):
        TrainerModelConfig.model_validate(model)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"ac": {"mode": "selective", "targets": ["@self_attn"]}}, "Invalid activation checkpointing target"),
        ({"ac": {"mode": "selective", "targets": ["aten::mm@"]}}, "Invalid activation checkpointing target"),
        ({"ac": {"mode": "selective", "targets": ["aten::mm@a@b"]}}, "Invalid activation checkpointing target"),
        ({"ac": {"mode": "selective", "targets": ["aten::mm@self_attn"]}, "compile": None}, "require model.compile"),
    ],
)
def test_invalid_ac_targets_fail(model, message):
    with pytest.raises(ValidationError, match=message):
        TrainerModelConfig.model_validate(model)


@pytest.mark.parametrize(("cp", "expected"), [(1, 262144), (8, 32768)])
def test_mega_moe_buffer_defaults_to_a_cp_shard(cp, expected):
    mega_moe = {"type": "mega_moe", "precision": "mxfp8"}
    rl = TrainerConfig.model_validate({"model": {"seq_len": 262144, "cp": cp, "moe": {"dispatch": mega_moe}}})
    sft = SFTConfig.model_validate(
        {"model": {"cp": cp, "moe": {"dispatch": mega_moe}}, "data": {"fake": {}, "seq_len": 262144}}
    )

    assert rl.model.moe.dispatch.max_tokens_per_rank == expected
    assert sft.model.moe.dispatch.max_tokens_per_rank == expected


def test_mega_moe_buffer_covers_mxfp8_padding():
    # A 10000-token pack pads to 10240 (a multiple of cp * 32), 1280 tokens per rank
    config = TrainerConfig.model_validate(
        {
            "model": {
                "seq_len": 10000,
                "cp": 8,
                "quantization": {"type": "mxfp8"},
                "moe": {"dispatch": {"type": "mega_moe", "precision": "mxfp8"}},
            }
        }
    )

    assert config.model.moe.dispatch.max_tokens_per_rank == 1280


def test_mega_moe_buffer_smaller_than_a_cp_shard_fails():
    with pytest.raises(ValidationError, match="must be at least the tokens a rank holds"):
        TrainerConfig.model_validate(
            {
                "model": {
                    "seq_len": 262144,
                    "cp": 8,
                    "moe": {"dispatch": {"type": "mega_moe", "max_tokens_per_rank": 16384}},
                }
            }
        )


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

    assert config.model.offload == "optimizer"


def test_freeze_moe_router_auto_resolves_per_trainer():
    """``freeze_moe_router='auto'`` (the default) freezes the router for RL, not SFT; explicit values are kept."""
    assert TrainerConfig.model_validate({}).model.freeze_moe_router is True
    assert SFTConfig.model_validate({}).model.freeze_moe_router is False

    for config_cls in (TrainerConfig, SFTConfig):
        for freeze in (True, False):
            assert config_cls.model_validate({"model": {"freeze_moe_router": freeze}}).model.freeze_moe_router is freeze


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_disables_gradient_clipping(config_cls):
    with pytest.warns(UserWarning, match="Gradient clipping prevents optimizer-in-backward"):
        config = config_cls.model_validate(
            {
                "model": {"offload": "full"},
                "optim": {"max_norm": 1.0},
            }
        )

    assert config.optim.max_norm is None


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_requires_supported_optimizer(config_cls):
    with pytest.raises(ValidationError, match="Full optimizer offload only supports AdamW and SignSGD"):
        config_cls.model_validate(
            {
                "model": {"offload": "full"},
                "optim": {"type": "muon", "max_norm": None},
            }
        )


@pytest.mark.parametrize("config_cls", [TrainerConfig, SFTConfig])
def test_full_optimizer_offload_accepts_sign_sgd(config_cls):
    config = config_cls.model_validate(
        {
            "model": {"offload": "full"},
            "optim": {"type": "sign_sgd", "max_norm": None},
        }
    )
    assert config.model.offload == "full"
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


@pytest.mark.parametrize(
    "config_class,source_args,expected_url,expected_key",
    [
        (
            EvalConfig,
            ["--source", '[{"env":{"id":"single_agent"}}]'],
            "https://configured.pinference.ai/api/v1",
            "PRIME_API_KEY",
        ),
        (
            SFTOnlineEvalConfig,
            ["--source", '[{"env":{"id":"single_agent"}}]'],
            "http://localhost:8000/v1",
            "VLLM_API_KEY",
        ),
        (OrchestratorModelConfig, [], "http://localhost:8000/v1", "VLLM_API_KEY"),
    ],
)
@pytest.mark.parametrize("overrides", [[], ["--client.skip-model-check"], ["--client.wait-for-ready-timeout", "10"]])
def test_client_defaults_survive_partial_overrides(
    monkeypatch, config_class, source_args, expected_url, expected_key, overrides
):
    monkeypatch.setenv("PRIME_INFERENCE_URL", "https://configured.pinference.ai/api/v1")
    config = cli(config_class, args=source_args + overrides)
    assert config.client.base_url == expected_url
    assert config.client.api_key_var == expected_key
    reloaded = config_class.model_validate_json(config.model_dump_json())
    assert reloaded.client == config.client


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


@pytest.mark.parametrize("group_token_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("source_token_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("source_token_limit", [200, None])
@pytest.mark.parametrize("nested_sampling", [False, True])
def test_sources_inherit_the_group_fields_they_leave_unset(
    group_token_key, source_token_key, source_token_limit, nested_sampling
):
    overrides = {source_token_key: source_token_limit, "top_k": 20}
    config = EvalConfig.model_validate(
        {
            "r": 4,
            "sampling": {
                group_token_key: 100,
                "temperature": 3.0,
                "reasoning_effort": "xhigh",
                "top_k": 40,
                "frequency_penalty": 0.5,
                "extra_body": {"a": 1},
            },
            "select": {"limit": 8, "include": {"idx": [":100"]}},
            "env": {"retries": {"max_retries": 3}},
            "source": [
                {"env": {"taskset": {"id": "gsm8k"}}},
                {
                    "name": "b",
                    "env": {"taskset": {"id": "gsm8k"}},
                    "group_size": 2,
                    "sampling": {
                        **({} if nested_sampling else overrides),
                        "reasoning_effort": "max",
                        "extra_body": {"b": 2, **(overrides if nested_sampling else {})},
                    },
                    "select": {"include": {"names": ["x"]}},
                },
            ],
        }
    )
    a, b = config.source
    assert (a.group_size, b.group_size) == (4, 2)
    assert isinstance(a.sampling, vf.SamplingConfig) and isinstance(b.sampling, vf.SamplingConfig)
    assert a.sampling.max_tokens == 100 and b.sampling.max_tokens == source_token_limit
    assert a.sampling.reasoning_effort == "xhigh" and a.sampling.top_k == 40
    assert b.sampling.model_dump(exclude_none=True) == {
        **({"max_tokens": source_token_limit} if source_token_limit is not None else {}),
        "temperature": 3.0,
        "reasoning_effort": "max",
        "top_k": 20,
        "frequency_penalty": 0.5,
        "a": 1,
        "b": 2,
    }
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

    assert [source.sampling.top_k for source in config.train.source] == [10000, 10000]
    assert all(source.sampling.extra_body["extra_args"]["prl_replay"] for source in config.train.source)
    data = config.model_dump(mode="json")
    data["eval"] = {
        "source": [
            {"name": "nucleus", "env": {"taskset": {"id": "reverse-text"}}, "sampling": {"top_p": 0.95}},
            {"name": "greedy", "env": {"taskset": {"id": "reverse-text"}}, "sampling": {"temperature": 0}},
        ]
    }
    mixed = OrchestratorConfig.model_validate(data)
    for source in mixed.eval.source:
        assert source.sampling.top_k == -1
        assert source.sampling.logprobs is None
    assert OrchestratorConfig.model_validate_json(mixed.model_dump_json()) == mixed


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "sampling,agent_sampling,rejected",
    [
        ({}, {}, False),
        ({"top_p": 0.9}, {"max_completion_tokens": 200, "reasoning_effort": "high"}, False),
        ({}, {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, False),
        ({}, {"temperature": 1.0}, True),
        ({}, {"temperature": None}, True),
        ({}, {"extra_body": {"top_k": 20}}, True),
        ({}, {"frequency_penalty": 0.5}, True),
        ({}, {"extra_body": {"frequency_penalty": 0.5}}, True),
        ({"extra_body": {"frequency_penalty": 0.5}}, {}, True),
    ],
)
def test_policy_sampling_is_validated_before_rollouts(sampling, agent_sampling, rejected, frozen):
    source = {
        "env": {"taskset": {"id": "reverse-text"}, "agent": {"sampling": agent_sampling}},
        "sampling": sampling,
    }
    if frozen:
        source["algo"] = {
            "type": "sft",
            "teacher": {"name": "teacher", "base_url": "http://localhost:8000/v1"},
        }
    if rejected and not frozen:
        with pytest.raises(ValidationError, match="sampling"):
            TrainSourceConfig.model_validate(source)
    else:
        config = TrainSourceConfig.model_validate(source)
        assert TrainSourceConfig.model_validate_json(config.model_dump_json()) == config


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
    assert config.trainer.weight_broadcast.inference_world_size == 2
    assert config.orchestrator.weight_broadcast.inference_world_size == 2
    assert config.inference.backend_port == 8101
    assert config.orchestrator.model.client.admin_base_url == ["http://localhost:8101/v1"]


def test_rl_launcher_gives_the_admin_token_to_callers_and_its_digest_to_checkers():
    from prime_rl.entrypoints.rl import admin_token_uses
    from prime_rl.utils.admin_token import admin_token_digest

    config = RLConfig.model_validate(
        {
            "trainer": {"control": {}},
            "orchestrator": {"control": {}},
            "inference": {"vllm": {"tensor_parallel_size": 1}},
            "deployment": {"type": "single_node", "gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
        }
    )
    uses = admin_token_uses(config)

    base, envs = uses.split({"HOME": "/home/user", "PRIME_RL_ADMIN_TOKEN_SHA256": "stale"})
    token = envs["orchestrator"]["PRIME_RL_ADMIN_TOKEN"]
    digest = admin_token_digest(token)
    assert len(token) >= 32
    assert base == {"HOME": "/home/user"}
    assert envs["inference"] == envs["trainer"] == {"PRIME_RL_ADMIN_TOKEN_SHA256": digest}
    assert envs["orchestrator"] == {"PRIME_RL_ADMIN_TOKEN": token, "PRIME_RL_ADMIN_TOKEN_SHA256": digest}

    base, envs = uses.split({"HOME": "/home/user", "PRIME_RL_ADMIN_TOKEN": "from-operator"})
    assert base == {"HOME": "/home/user"}
    assert envs["orchestrator"]["PRIME_RL_ADMIN_TOKEN"] == "from-operator"
    assert envs["inference"] == {"PRIME_RL_ADMIN_TOKEN_SHA256": admin_token_digest("from-operator")}


def test_sft_launcher_gives_the_admin_token_to_online_evals_and_its_digest_to_checkers():
    from prime_rl.entrypoints.launcher import admin_token_uses
    from prime_rl.utils.admin_token import admin_token_digest

    config = SFTConfig.model_validate(
        {
            "control": {},
            "eval": {"source": [{"name": "reverse-text", "env": {"taskset": {"id": "reverse-text"}}}]},
            "inference": {},
            "deployment": {"num_train_gpus": 1, "num_infer_gpus": 1},
        }
    )

    base, envs = admin_token_uses(config).split({"HOME": "/home/user"})
    token = envs["online-eval"]["PRIME_RL_ADMIN_TOKEN"]
    assert base == {"HOME": "/home/user"}
    assert envs["online-eval"] == {"PRIME_RL_ADMIN_TOKEN": token}
    assert envs["inference"] == envs["trainer"] == {"PRIME_RL_ADMIN_TOKEN_SHA256": admin_token_digest(token)}


def test_single_node_nccl_resolved_json_roundtrips(tmp_path):
    """The resolved rl.json re-parses (as a SLURM launch does) to the same config."""
    import json

    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"vllm": {"tensor_parallel_size": 1}},
            "deployment": {"type": "single_node", "gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
        }
    )
    path = tmp_path / "rl.json"
    path.write_text(json.dumps(dump_resolved_config(config, exclude={"slurm", "dry_run", "clean"})))
    reloaded = cli(RLConfig, args=["@", str(path)])
    assert reloaded.trainer.weight_broadcast.inference_world_size == 2
    assert reloaded == config


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"orchestrator": {"pad_to_multiple_of": 4}}, "orchestrator.pad_to_multiple_of"),
        ({"orchestrator": {"num_train_workers": 3}}, "orchestrator.num_train_workers"),
        ({"orchestrator": {"weight_broadcast": {"type": "filesystem"}}}, "orchestrator.weight_broadcast"),
        ({"trainer": {"weight_broadcast": {"type": "nccl", "timeout": 60}}}, "trainer.weight_broadcast"),
    ],
)
def test_rl_rejects_conflicting_launcher_owned_fields(overrides, match):
    data = {
        "trainer": {},
        "orchestrator": {},
        "inference": {"vllm": {"tensor_parallel_size": 1}},
        "deployment": {"type": "single_node", "gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
    }
    for key, value in overrides.items():
        data[key] = value
    with pytest.raises(ValidationError, match=match):
        RLConfig.model_validate(data)


def test_rl_nccl_broadcast_needs_two_gpus():
    with pytest.raises(ValidationError, match="at least 2 GPUs"):
        RLConfig.model_validate(
            {
                "trainer": {"data": {"fake": {}}},
                "orchestrator": {},
                "weight_broadcast": {"type": "nccl"},
                "deployment": {"type": "single_node", "gpus_per_node": 1, "num_train_gpus": 1, "num_infer_gpus": 0},
            }
        )


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


def test_disaggregated_inference_inherits_deployment_gpus_per_node():
    config = {
        "trainer": {},
        "orchestrator": {},
        "inference": {"vllm": {"tensor_parallel_size": 2}, "deployment": {"type": "disaggregated"}},
        "deployment": {"type": "multi_node", "gpus_per_node": 4, "num_train_nodes": 1},
        "slurm": {},
    }
    resolved = RLConfig.model_validate(config)
    assert resolved.inference.deployment.gpus_per_node == 4
    assert resolved.inference.vllm.data_parallel_size_local == 2

    config["deployment"].pop("gpus_per_node")
    config["inference"]["deployment"]["gpus_per_node"] = 4
    with pytest.raises(ValidationError, match="gpus_per_node must match"):
        RLConfig.model_validate(config)


def test_ac_none_disables_ac_offloading():
    config = TrainerModelConfig.model_validate({"ac": None})
    assert config.ac is None
    assert config.ac_offloading is None


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
    enabled; a shared ``[monitors.prime]`` reaches both."""
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
    assert config.trainer.monitors.prime.name == config.orchestrator.monitors.prime.name == "shared-prime"


@pytest.mark.parametrize(
    "monitor,trainer,orchestrator",
    [("wandb", {"wandb": {"project": "p"}}, {}), ("prime", {"prime": {}}, {})],
)
def test_monitor_on_in_one_component_only_is_rejected(monitor, trainer, orchestrator):
    """The components log to one run per monitor, so a monitor is on in all or none."""
    with pytest.raises(ValueError, match=rf"monitors.{monitor} must match across components"):
        RLConfig.model_validate(
            {
                "model": {"name": "Qwen/Qwen3-0.6B"},
                "trainer": {"monitors": trainer},
                "orchestrator": {"renderer": {"name": "default"}, "monitors": orchestrator},
            }
        )


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
            "data": {"fake": {}},
            "renderer": {"name": "default"},
        }
    )
    assert config.renderer.name == "default"


def test_sft_source_columns_override_data_columns():
    config = SFTDataConfig.model_validate(
        {
            "columns": {"tools": "schemas"},
            "source": [{"name": "a", "dataset": "x", "columns": {"messages": "chat"}}, {"name": "b", "dataset": "y"}],
        }
    )
    assert [(source.columns.messages, source.columns.tools) for source in config.source] == [
        ("chat", "schemas"),
        ("messages", "schemas"),
    ]


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ([{"name": "a", "dataset": "x"}, {"name": "a", "dataset": "y"}], "names must be unique"),
        ([{"name": "a", "dataset": "x", "weight": 1}, {"name": "b", "dataset": "y"}], "on every source or on none"),
        ([{"name": "a", "dataset": "x", "weight": 0}], "sum to more than 0"),
    ],
)
def test_sft_sources_reject_invalid_mixtures(sources, message):
    with pytest.raises(ValidationError, match=message):
        SFTDataConfig.model_validate({"source": sources})


def test_sft_val_rejects_fake_data():
    with pytest.raises(ValidationError, match="not fake data"):
        SFTConfig.model_validate({"val": {"data": {"fake": {}}}})


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


def test_pd_role_connectors():
    mooncake = {"type": "mooncake", "cpu": {"num_bytes": 1000}}
    deployment = {"type": "disaggregated", "decode_kv_cache_offload": {**mooncake, "cpu": {"num_bytes": 0}}}
    config = InferenceConfig(deployment=deployment, slurm={}, kv_cache_offload=mooncake)
    prefill = config.for_pd_role("prefill").build_kv_transfer_config()["kv_connector_extra_config"]["connectors"]
    decode = config.for_pd_role("decode").build_kv_transfer_config()["kv_connector_extra_config"]["connectors"]
    assert [c["kv_connector"] for c in prefill] == ["NixlConnector", "MooncakeStoreConnector"]
    assert [c["kv_connector"] for c in decode] == ["NixlConnector", "MooncakeStoreConnector"]
    # The store lookup runs off the scheduler thread.
    assert prefill[1]["kv_connector_extra_config"]["lookup_async"] is True
    assert decode[1]["kv_connector_extra_config"]["save_decode_cache"]

    with pytest.raises(ValidationError, match="decode_kv_cache_offload"):
        InferenceConfig(deployment=deployment, slurm={}, kv_cache_offload=mooncake, hisparse={"host_pool_gib": 160})
    deployment["decode_kv_cache_offload"] = None
    config = InferenceConfig(
        deployment=deployment, slurm={}, kv_cache_offload=mooncake, hisparse={"host_pool_gib": 160}
    )
    decode = config.for_pd_role("decode").build_kv_transfer_config()["kv_connector_extra_config"]["connectors"]
    assert [c["kv_connector"] for c in decode] == ["NixlConnector", "HiSparseConnector"]
    assert config.for_pd_role("prefill").hisparse is None

    with pytest.raises(ValidationError, match="native KV cache offload"):
        InferenceConfig(
            kv_cache_offload={"type": "native", "cpu": {"num_bytes": 1000}}, hisparse={"host_pool_gib": 160}
        )


def test_pd_bidirectional_kv_transfer():
    deployment = {"type": "disaggregated", "bidirectional_kv_transfer": {"decode_blocks_ttl_s": 600}}
    config = InferenceConfig(deployment=deployment, slurm={})
    assert config.deployment.bidirectional_kv_transfer.effective_router_ttl_s == 570

    for role in ("prefill", "decode"):
        connector = config.for_pd_role(role).build_kv_transfer_config()
        assert connector["kv_connector"] == "NixlConnector"
        assert connector["kv_connector_extra_config"] == {
            "num_threads": 1,
            "bidirectional_kv_xfer": True,
            "decoder_kv_blocks_ttl": 600,
        }

    # The per-role engine configs drop `deployment`; bidirectional transfer survives through the persisted TTL.
    role = InferenceConfig(**dump_resolved_config(config.for_pd_role("decode"), exclude={"deployment", "slurm"}))
    assert role.build_kv_transfer_config()["kv_connector_extra_config"]["decoder_kv_blocks_ttl"] == 600

    with pytest.raises(ValidationError, match="shorter than decode_blocks_ttl_s"):
        InferenceConfig(
            deployment={**deployment, "bidirectional_kv_transfer": {"decode_blocks_ttl_s": 60, "router_ttl_s": 60}},
            slurm={},
        )
    with pytest.raises(ValidationError, match="requires the vllm-router"):
        InferenceConfig(deployment=deployment, slurm={}, router={"type": "llm-d"})
    with pytest.raises(ValidationError, match="HiSparse"):
        InferenceConfig(deployment=deployment, slurm={}, hisparse={"host_pool_gib": 160})

    # One-way P/D is unchanged: no bidirectional flags.
    config = InferenceConfig(deployment={"type": "disaggregated"}, slurm={})
    decode = config.for_pd_role("decode").build_kv_transfer_config()
    assert decode["kv_connector_extra_config"] == {"num_threads": 1}


def test_pd_spill_to_store_runs_the_store_connector_in_capacity_only_mode():
    store = {"type": "mooncake", "cpu": {"num_bytes": 1000}}
    deployment = {
        "type": "disaggregated",
        "bidirectional_kv_transfer": {"decode_blocks_ttl_s": 20, "spill_to_store": {}},
    }
    config = InferenceConfig(deployment=deployment, kv_cache_offload=store, slurm={})
    # The router keeps handles past the lease, so a late turn reaches prefill and is fetched from the store.
    assert config.deployment.bidirectional_kv_transfer.effective_router_ttl_s == 3600

    for role in ("prefill", "decode"):
        role_config = InferenceConfig(**dump_resolved_config(config.for_pd_role(role), exclude={"deployment", "slurm"}))
        nixl, mooncake = role_config.build_kv_transfer_config()["kv_connector_extra_config"]["connectors"]
        assert nixl["kv_connector_extra_config"] == {
            "num_threads": 1,
            "bidirectional_kv_xfer": True,
            "decoder_kv_blocks_ttl": 20,
            "prime_rl_spill_lead_s": 10.0,
            "prime_rl_spill_fetch_wait_s": 6.0,
        }
        # vLLM's store connector only mounts the client: no lookups, no loads, no saves (not even decode's).
        assert mooncake == {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {"lookup_async": True, "enable_lookup": False},
        }

    with pytest.raises(ValidationError, match="needs a Mooncake kv_cache_offload"):
        InferenceConfig(deployment=deployment, slurm={})
    with pytest.raises(ValidationError, match="longer than"):
        InferenceConfig(
            deployment={
                **deployment,
                "bidirectional_kv_transfer": {"decode_blocks_ttl_s": 10, "spill_to_store": {}},
            },
            kv_cache_offload=store,
            slurm={},
        )

    # Without the spill the store connector keeps its usual role under P/D.
    plain = InferenceConfig(
        deployment={"type": "disaggregated", "bidirectional_kv_transfer": {"decode_blocks_ttl_s": 480}},
        kv_cache_offload=store,
        slurm={},
    )
    mooncake = plain.for_pd_role("decode").build_kv_transfer_config()["kv_connector_extra_config"]["connectors"][1]
    assert mooncake["kv_role"] == "kv_both" and mooncake["kv_connector_extra_config"]["save_decode_cache"] is True
    assert "enable_lookup" not in mooncake["kv_connector_extra_config"]


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
                "data": {"source": [{"name": "reverse", "dataset": "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"}]},
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


@pytest.mark.parametrize("top_k", [32, 256])
@pytest.mark.parametrize("top_p", [0.9, 1.0])
def test_score_centering_captures_complete_bounded_support(top_k, top_p):
    config = RLConfig.model_validate(
        {
            "trainer": {"loss": {"type": "icepop", "score_centering": True}},
            "orchestrator": {
                "train": {
                    "source": [
                        {
                            "env": {"taskset": {"id": "reverse-text"}},
                            "sampling": {"top_k": top_k, "top_p": top_p},
                        }
                    ]
                }
            },
            "inference": {},
        }
    )
    assert config.orchestrator.train.source[0].sampling.logprobs == 1
    assert config.inference.enable_return_sampling_mask
    assert RLConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    ("config_cls", "default_type"),
    [(PretrainConfig, "scratch"), (SFTConfig, "pretrained"), (TrainerConfig, "pretrained")],
)
def test_model_init_type_defaults_per_entrypoint(tmp_path, config_cls, default_type: str):
    data = {"data": {"source": [{"name": "tokens", "path": str(tmp_path)}]}} if config_cls is PretrainConfig else {}
    assert config_cls.model_validate(data).model.init.type == default_type
    # A [model.init] table without a type keeps the entrypoint's default.
    assert config_cls.model_validate({**data, "model": {"init": {}}}).model.init.type == default_type


def test_model_init_seed_belongs_to_scratch():
    assert SFTConfig.model_validate({"model": {"init": {"type": "scratch", "seed": 1}}}).model.init.seed == 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SFTConfig.model_validate({"model": {"init": {"seed": 1}}})


@pytest.mark.parametrize("init_type", ["hf", "ckpt"])
def test_model_init_types_of_earlier_versions_fail(init_type: str):
    with pytest.raises(ValidationError, match="does not match any of the expected tags: 'scratch', 'pretrained'"):
        SFTConfig.model_validate({"model": {"init": {"type": init_type}}})


def test_model_init_scratch_rejects_lora():
    with pytest.raises(ValidationError, match='model.init.type = "scratch" does not load'):
        SFTConfig.model_validate({"model": {"init": {"type": "scratch"}, "lora": {}}})


@pytest.mark.parametrize("tokenizer", [None, "Qwen/Qwen3-0.6B"])
def test_inference_parsers_resolve_from_the_tokenizer_name(tokenizer: str | None):
    data = {
        "model": {"name": "runs/sft/checkpoints/step_500"},
        "trainer": {},
        "orchestrator": {
            "train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}}]},
            "renderer": {"name": "qwen3"},
        },
        "inference": {},
    }
    if tokenizer is None:
        with pytest.warns(UserWarning, match="set tokenizer.name"):
            config = RLConfig.model_validate(data)
        assert config.inference.vllm.tool_call_parser is None
    else:
        config = RLConfig.model_validate({**data, "tokenizer": {"name": tokenizer}})
        assert config.inference.vllm.tool_call_parser == "hermes"


def test_rl_shared_model_init_reaches_the_trainer():
    config = RLConfig.model_validate(
        {
            "model": {"init": {"type": "scratch", "seed": 3}},
            "trainer": {},
            "orchestrator": {
                "train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}}]},
                "renderer": {"name": "qwen3"},
            },
        }
    )
    assert config.trainer.model.init.type == "scratch" and config.trainer.model.init.seed == 3


@pytest.mark.parametrize(
    ("init_type", "explicit", "load_format"),
    [("pretrained", None, None), ("scratch", None, "dummy"), ("scratch", "auto", "auto")],
)
def test_inference_loads_dummy_weights_for_a_scratch_init(
    init_type: str, explicit: str | None, load_format: str | None
):
    vllm = {"speculative_config": {"method": "eagle3", "model": "drafts/eagle3"}}
    if explicit is not None:
        vllm["load_format"] = explicit
    config = RLConfig.model_validate(
        {
            "model": {"init": {"type": init_type}},
            "trainer": {},
            "orchestrator": {
                "train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}}]},
                "renderer": {"name": "qwen3"},
            },
            "inference": {"vllm": vllm},
        }
    )
    namespace = config.inference.to_namespace()
    assert getattr(namespace, "load_format", None) == load_format
    draft_load_config = {"load_format": "auto"} if load_format == "dummy" else None
    assert namespace.speculative_config.get("draft_load_config") == draft_load_config
    assert RLConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    ("init_type", "vllm", "match"),
    [
        ("scratch", {"speculative_config": {"method": "mtp"}}, "reads its draft weights from model.name"),
        ("scratch", {"speculative_config": {"model": "drafts/qwen3-0.6b"}}, "with the target's load format"),
        ("pretrained", {"load_format": "dummy", "speculative_config": {"method": "mtp"}}, "reads its draft weights"),
    ],
)
def test_a_draft_without_its_own_weights_fails_under_a_dummy_target(init_type: str, vllm: dict, match: str):
    with pytest.raises(ValidationError, match=match):
        RLConfig.model_validate(
            {
                "model": {"init": {"type": init_type}},
                "trainer": {},
                "orchestrator": {
                    "train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}}]},
                    "renderer": {"name": "qwen3"},
                },
                "inference": {"vllm": vllm},
            }
        )


def test_score_centering_replays_top_p_alone():
    config = RLConfig.model_validate(
        {
            "trainer": {"loss": {"type": "icepop", "score_centering": True}},
            "orchestrator": {
                "train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}, "sampling": {"top_p": 0.95}}]}
            },
            "inference": {},
        }
    )
    sampling = config.orchestrator.train.source[0].sampling
    assert sampling.top_k == 10000 and sampling.logprobs == 1
    assert sampling.extra_body["extra_args"]["prl_replay"]
    assert config.inference.enable_return_sampling_mask


def test_rl_nvfp4_experts_set_up_matching_engines():
    def rl(vllm, env_vars=None):
        return RLConfig.model_validate(
            {
                "trainer": {"model": {"moe": {"compute": {"type": "nvfp4", "four_over_six": True}}}},
                "orchestrator": {},
                "inference": {"vllm": {"tensor_parallel_size": 1, **vllm}, "env_vars": env_vars or {}},
                "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
            }
        )

    nvfp4 = {"quantization": "online", "quantization_config": {"linear": "mxfp8", "moe": "nvfp4_per_token"}}
    config = rl(nvfp4)
    assert config.inference.env_vars["FLASHINFER_NVFP4_4OVER6"] == "1"
    assert config.inference.env_vars["FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"] == "1"
    assert config.inference.vllm.moe_backend == "flashinfer_trtllm"
    with pytest.raises(ValueError, match="moe_backend"):
        rl({**nvfp4, "moe_backend": "flashinfer_cutedsl"})
    with pytest.raises(ValueError, match="nvfp4_per_token"):
        rl({"quantization": "mxfp8"})
    with pytest.raises(ValueError, match="FLASHINFER_NVFP4_4OVER6"):
        rl(nvfp4, {"FLASHINFER_NVFP4_4OVER6": "0"})


@pytest.mark.parametrize("rollouts", [False, True])
def test_shared_ckpt_rollouts_reaches_the_orchestrator(rollouts):
    config = RLConfig.model_validate(
        {
            "ckpt": {"interval": 5, "rollouts": rollouts},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.orchestrator.ckpt.rollouts is rollouts
    assert config.orchestrator.ckpt.interval == config.trainer.ckpt.interval == 5
    assert RLConfig.model_validate_json(config.model_dump_json()) == config


def test_ckpt_rollouts_is_off_by_default():
    config = RLConfig.model_validate({"ckpt": {}, "trainer": {}, "orchestrator": {"renderer": {"name": "default"}}})
    assert config.orchestrator.ckpt.rollouts is False


def test_shared_ckpt_reaches_trainer_and_orchestrator():
    config = RLConfig.model_validate(
        {
            "ckpt": {"output_dir": "/ckpts", "keep_last": 3, "keep_interval": 100},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    for ckpt in (config.trainer.ckpt, config.orchestrator.ckpt):
        assert (ckpt.output_dir, ckpt.keep_last, ckpt.keep_interval) == (Path("/ckpts"), 3, 100)


@pytest.mark.parametrize("leaf", [("output_dir", "/ckpts"), ("keep_last", 3), ("keep_interval", 100)])
def test_trainer_and_orchestrator_share_one_ckpt_dir_and_policy(leaf):
    name, value = leaf
    with pytest.raises(ValueError, match=f"ckpt.{name}"):
        RLConfig.model_validate(
            {"ckpt": {}, "trainer": {"ckpt": {name: value}}, "orchestrator": {"renderer": {"name": "default"}}}
        )


def test_env_worker_pool_uses_ckpt_output_dir_as_save_root(tmp_path):
    from prime_rl.entrypoints.rl import write_subconfigs

    ckpt_dir = tmp_path / "external_ckpts"
    config = RLConfig.model_validate(
        {
            "output_dir": str(tmp_path / "run"),
            "ckpt": {"output_dir": str(ckpt_dir), "rollouts": True},
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "default"},
                "train": {"source": [{"env": {"taskset": {"id": "harbor"}}}]},
            },
        }
    )
    cfg_dir = tmp_path / "configs"
    write_subconfigs(config, cfg_dir)
    pool_configs = list((cfg_dir / "envs" / "train").glob("*.json"))
    assert pool_configs
    data = json.loads(pool_configs[0].read_text())
    assert Path(data["output_dir"]) == ckpt_dir.resolve()
