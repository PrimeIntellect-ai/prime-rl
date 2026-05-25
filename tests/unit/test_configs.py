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
    """Any TOML file inside `configs/` or `examples/` (skips the configs/private/ submodule)."""
    private = Path("configs/private")
    config_files = [p for p in Path("configs").rglob("*.toml") if private not in p.parents]
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


def test_orchestrator_vlm_requires_renderer():
    with pytest.raises(ValidationError, match="orchestrator.use_renderer must be true when model.vlm is set"):
        OrchestratorConfig.model_validate(
            {
                "student": {
                    "model": {
                        "name": "Qwen/Qwen3-VL-4B-Instruct",
                        "vlm": {
                            "vision_encoder_attr": "model.visual",
                            "language_model_attr": "model.language_model",
                        },
                    }
                },
                "use_renderer": False,
            }
        )

    config = OrchestratorConfig.model_validate(
        {
            "student": {
                "model": {
                    "name": "Qwen/Qwen3-VL-4B-Instruct",
                    "vlm": {
                        "vision_encoder_attr": "model.visual",
                        "language_model_attr": "model.language_model",
                    },
                }
            },
        }
    )

    assert config.use_renderer is True


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_shared_model_name_propagates_to_subconfigs():
    model_name = "PrimeIntellect/test-model"
    config = RLConfig.model_validate(
        {
            "model": {"name": model_name},
            "trainer": {},
            "orchestrator": {"use_renderer": False},
            "inference": {},
        }
    )
    assert config.trainer.model.name == model_name
    assert config.orchestrator.student.model.name == model_name
    assert config.inference is not None and config.inference.model.name == model_name
    assert config.trainer.tokenizer.name == model_name
    assert config.orchestrator.tokenizer.name == model_name


def test_shared_tokenizer_propagates_when_subconfigs_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"name": "my-tokenizer"},
            "trainer": {},
            "orchestrator": {"use_renderer": False},
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
                "orchestrator": {"use_renderer": False},
            }
        )


def test_tokenizer_name_falls_back_to_model_name_when_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"trust_remote_code": True},
            "trainer": {},
            "orchestrator": {"use_renderer": False},
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
    ``orchestrator.student.model.name`` after propagation, silently overriding
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
                "use_renderer": False,
                "tokenizer": {"name": "explicit-orch-tok"},
            },
        }
    )
    # Shared model.name reached every sub-config that didn't override it.
    assert config.trainer.model.name == "M"
    assert config.orchestrator.student.model.name == "M"
    # Trainer didn't specify a tokenizer, so it falls back to the propagated model name.
    assert config.trainer.tokenizer.name == "M"
    # Orchestrator's explicit tokenizer name survived.
    assert config.orchestrator.tokenizer.name == "explicit-orch-tok"


def test_tokenizer_chat_template_mismatch_raises():
    with pytest.raises(ValidationError, match="chat_template"):
        RLConfig.model_validate(
            {
                "trainer": {"tokenizer": {"chat_template": "A"}},
                "orchestrator": {"use_renderer": False, "tokenizer": {"chat_template": "B"}},
            }
        )


def test_shared_seq_len_propagates_to_subconfigs():
    config = RLConfig.model_validate(
        {
            "seq_len": 4096,
            "trainer": {},
            "orchestrator": {"use_renderer": False},
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
                "orchestrator": {"use_renderer": False},
            }
        )


def test_shared_and_sub_model_name_conflict_raises():
    """Setting model.name at the shared level and on a sub-config is a conflict."""
    with pytest.raises(ValidationError, match=r"model.name.*trainer.model.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "X"},
                "trainer": {"model": {"name": "Y"}},
                "orchestrator": {"use_renderer": False},
            }
        )


def test_shared_and_sub_max_steps_conflict_raises():
    """Top-level scalar shared fields also participate in the mutex check."""
    with pytest.raises(ValidationError, match=r"max_steps.*orchestrator.max_steps"):
        RLConfig.model_validate(
            {
                "max_steps": 100,
                "trainer": {},
                "orchestrator": {"use_renderer": False, "max_steps": 200},
            }
        )


def test_trainer_chat_template_cascades_to_inference():
    """``[trainer.tokenizer] chat_template`` set directly (no shared
    ``[tokenizer] chat_template``) must still reach
    ``inference.model.chat_template`` so vLLM's ``--chat-template`` is wired
    up. Regression: the original ``auto_setup_tokenizer`` cascaded this; the
    refactored propagator must keep doing it."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"tokenizer": {"chat_template": "TPL"}},
            "orchestrator": {"use_renderer": False, "tokenizer": {"chat_template": "TPL"}},
            "inference": {},
        }
    )
    assert config.trainer.tokenizer.chat_template == "TPL"
    assert config.orchestrator.tokenizer.chat_template == "TPL"
    assert config.inference is not None
    assert config.inference.model.chat_template == "TPL"


def test_shared_wandb_fields_propagate_to_subconfigs():
    """Every ``SharedWandbConfig`` leaf (project, entity, group, tags, offline)
    propagates to both trainer.wandb and orchestrator.wandb, not just project
    and offline. Regression for a miss in the inline propagator."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "wandb": {
                "project": "shared-proj",
                "entity": "shared-entity",
                "group": "shared-group",
                "tags": ["a", "b"],
                "shared": False,
                "offline": True,
            },
            "trainer": {},
            "orchestrator": {"use_renderer": False},
        }
    )
    for component in (config.trainer.wandb, config.orchestrator.wandb):
        assert component is not None
        assert component.project == "shared-proj"
        assert component.entity == "shared-entity"
        assert component.group == "shared-group"
        assert component.tags == ["a", "b"]
        assert component.offline is True


def test_empty_shared_ckpt_block_does_not_conflict_with_subconfig_ckpt():
    """An empty shared [ckpt] block is a presence-only signal, not a field
    setting — it should not conflict with a non-empty [trainer.ckpt]."""
    config = RLConfig.model_validate(
        {
            "ckpt": {},  # empty block, no field set
            "trainer": {"ckpt": {"interval": 50}},
            "orchestrator": {"use_renderer": False, "ckpt": {"interval": 50}},
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
            "trainer": {"model": {"impl": "custom"}},
            "orchestrator": {"use_renderer": False},
        }
    )
    assert config.trainer.model.name == "Qwen/Qwen3-0.6B"
    assert config.trainer.model.impl == "custom"


def test_shared_output_dir_propagates_through_cli(tmp_path):
    """Shared output_dir from CLI reaches sub-configs even when tyro constructs sub-configs before the before-validator."""
    toml_path = tmp_path / "cfg.toml"
    write_toml(
        toml_path,
        {
            "max_steps": 1,
            "seq_len": 128,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {},
            "orchestrator": {"batch_size": 16, "group_size": 1},
            "inference": {},
        },
    )
    shared_out = tmp_path / "shared"
    config = cli(RLConfig, args=["@", str(toml_path), "--output-dir", str(shared_out)])
    assert config.trainer.output_dir == shared_out
    assert config.orchestrator.output_dir == shared_out / "run_default"


def test_orchestrator_renderer_auto_rejects_unmapped_model():
    """use_renderer=True with renderer.settings.name='auto' must reject models not in MODEL_RENDERER_MAP."""
    with pytest.raises(ValidationError, match="silently fall back to DefaultRenderer"):
        OrchestratorConfig.model_validate({"model": {"name": "not-a-real-org/not-a-real-model"}})


def test_orchestrator_renderer_auto_accepts_mapped_model():
    """The default Qwen model is in MODEL_RENDERER_MAP and should validate cleanly."""
    config = OrchestratorConfig.model_validate({"model": {"name": "Qwen/Qwen3-0.6B"}})
    assert config.use_renderer is True
    assert config.renderer.settings.name == "auto"


def test_orchestrator_explicit_renderer_skips_unmapped_check():
    """Explicit renderer.settings.name bypasses the auto-resolution check — user opted in."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"settings": {"name": "qwen3"}},
        }
    )
    assert config.renderer.settings.name == "qwen3"


def test_orchestrator_use_renderer_false_skips_unmapped_check():
    """use_renderer=False means the renderer client isn't used, so MODEL_RENDERER_MAP doesn't apply."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "use_renderer": False,
        }
    )
    assert config.use_renderer is False


def test_orchestrator_explicit_default_renderer_with_unmapped_model():
    """renderer.settings.name='default' is an explicit opt-in to DefaultRenderer and must pass."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"settings": {"name": "default", "tool_parser": "qwen3"}},
        }
    )
    assert config.renderer.settings.name == "default"
    assert config.renderer.settings.tool_parser == "qwen3"


def test_shared_model_name_resolves_inference_parsers():
    """Shared [model] name must reach inference.model BEFORE ModelConfig's after-validator
    runs auto_resolve_parsers — i.e. the parsers resolve from the propagated name, not
    from an empty default.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"use_renderer": False},
            "inference": {},
        }
    )
    assert config.inference is not None
    assert config.inference.model.name == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config.inference.model.tool_call_parser == "qwen3_coder"


def test_explicit_inference_parser_wins_over_auto():
    """Explicit inference.model.tool_call_parser is preserved even when the shared model
    name would otherwise auto-resolve to something else."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"use_renderer": False},
            "inference": {"model": {"tool_call_parser": "hermes"}},
        }
    )
    assert config.inference is not None
    assert config.inference.model.tool_call_parser == "hermes"
