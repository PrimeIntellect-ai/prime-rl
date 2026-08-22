from pathlib import Path

import pytest

from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.monitors import (
    MonitorsConfig,
    OrchestratorMonitorsConfig,
    PrimeMonitorConfig,
    WandbMonitorConfig,
)
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.shared import PROTECTED_ENV_VARS, MetricsServerConfig, reject_protected_env_vars
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.platform_env import ComponentConfig, ComponentRole, apply_platform_env


def component_config(role: ComponentRole) -> ComponentConfig:
    config_type = {
        "orchestrator": OrchestratorConfig,
        "trainer": TrainerConfig,
        "inference": InferenceConfig,
        "env-server": EnvServerConfig,
    }[role]
    return config_type.model_construct()


@pytest.mark.parametrize("role", ["orchestrator", "trainer", "inference", "env-server"])
def test_common_overrides_apply_to_every_component(role: ComponentRole):
    config = component_config(role)
    config.log.json_logging = False
    config.output_dir = Path("from-config")

    changes = apply_platform_env(
        config,
        role,
        {
            "PRL_JSON_LOGGING": "1",
            "PRL_RUN_ID": "run-123",
            "PRL_OUTPUT_DIR": "/mnt/run-123",
        },
    )

    assert config.log.json_logging is True
    assert config.output_dir == Path("/mnt/run-123")
    assert len(changes) == 2
    assert all("overrides" in change for change in changes)


def test_orchestrator_force_flags_enable_missing_monitors():
    config = OrchestratorConfig.model_construct(monitors=OrchestratorMonitorsConfig(prime=None, wandb=None))
    environ = {
        "PRL_FORCE_PRIME_MONITOR": "1",
        "PRL_FORCE_WANDB_MONITOR": "1",
        "PRL_RUN_ID": "run-123",
    }

    changes = apply_platform_env(config, "orchestrator", environ)

    assert isinstance(config.monitors.prime, PrimeMonitorConfig)
    assert isinstance(config.monitors.wandb, WandbMonitorConfig)
    assert environ["WANDB_SHARED_MODE"] == "1"
    assert environ["WANDB_RUN_ID"] == "run-123"
    assert environ["WANDB_SHARED_LABEL"] == "orchestrator"
    assert len(changes) == 3


def test_force_flags_preserve_existing_monitor_config():
    prime = PrimeMonitorConfig(name="prime-name")
    wandb = WandbMonitorConfig(
        project="custom-project",
        entity="custom-entity",
        name="wandb-name",
        group="custom-group",
        tags=["custom-tag"],
    )
    config = OrchestratorConfig.model_construct(
        monitors=OrchestratorMonitorsConfig(prime=prime, wandb=wandb),
    )
    environ = {
        "PRL_FORCE_PRIME_MONITOR": "1",
        "PRL_FORCE_WANDB_MONITOR": "1",
        "PRL_RUN_ID": "run-123",
        "WANDB_RUN_ID": "wrong-run",
        "WANDB_SHARED_LABEL": "wrong-role",
        "WANDB_SHARED_MODE": "0",
    }

    apply_platform_env(config, "orchestrator", environ)

    assert config.monitors.prime is prime
    assert config.monitors.wandb is wandb
    assert wandb.project == "custom-project"
    assert wandb.entity == "custom-entity"
    assert wandb.name == "wandb-name"
    assert wandb.group == "custom-group"
    assert wandb.tags == ["custom-tag"]
    assert environ["WANDB_RUN_ID"] == "run-123"
    assert environ["WANDB_SHARED_LABEL"] == "orchestrator"
    assert environ["WANDB_SHARED_MODE"] == "1"


def test_force_wandb_preserves_offline_monitor_semantics():
    wandb = WandbMonitorConfig(offline=True)
    config = TrainerConfig.model_construct(monitors=MonitorsConfig(wandb=wandb))
    environ = {
        "PRL_FORCE_WANDB_MONITOR": "1",
        "PRL_RUN_ID": "run-123",
    }

    changes = apply_platform_env(config, "trainer", environ)

    assert config.monitors.wandb is wandb
    assert "WANDB_SHARED_MODE" not in environ
    assert "WANDB_RUN_ID" not in environ
    assert "WANDB_SHARED_LABEL" not in environ
    assert changes == []


@pytest.mark.parametrize(
    ("metrics_server", "expected_host"),
    [
        (None, "0.0.0.0"),
        (MetricsServerConfig(port=9000, host="127.0.0.1"), "127.0.0.1"),
    ],
)
def test_metrics_port_enables_server_and_preserves_host(metrics_server, expected_host: str):
    config = TrainerConfig.model_construct(metrics_server=metrics_server)

    changes = apply_platform_env(config, "trainer", {"PRL_METRICS_PORT": "8123"})

    assert config.metrics_server == MetricsServerConfig(port=8123, host=expected_host)
    assert len(changes) == 1
    assert "$PRL_METRICS_PORT" in changes[0]


@pytest.mark.parametrize("value", ["", "0", "65536", "abc", " 8123 "])
def test_metrics_port_rejects_invalid_values(value: str):
    config = TrainerConfig.model_construct()

    with pytest.raises(ValueError, match="PRL_METRICS_PORT"):
        apply_platform_env(config, "trainer", {"PRL_METRICS_PORT": value})


@pytest.mark.parametrize(
    ("role", "name"),
    [
        ("inference", "PRL_JSON_LOGGING"),
        ("orchestrator", "PRL_FORCE_PRIME_MONITOR"),
        ("trainer", "PRL_FORCE_WANDB_MONITOR"),
    ],
)
@pytest.mark.parametrize("value", ["", "0", "true"])
def test_flags_only_accept_one(role: ComponentRole, name: str, value: str):
    with pytest.raises(ValueError, match=name):
        apply_platform_env(component_config(role), role, {name: value})


@pytest.mark.parametrize("name", ["PRL_RUN_ID", "PRL_OUTPUT_DIR"])
def test_shared_string_values_reject_empty_or_whitespace(name: str):
    with pytest.raises(ValueError, match=name):
        apply_platform_env(component_config("inference"), "inference", {name: "  "})


def test_role_specific_values_are_ignored_by_unrelated_components():
    config = InferenceConfig.model_construct()

    changes = apply_platform_env(
        config,
        "inference",
        {
            "PRL_FORCE_PRIME_MONITOR": "invalid-but-unrelated",
            "PRL_FORCE_WANDB_MONITOR": "invalid-but-unrelated",
            "PRL_METRICS_PORT": "invalid-but-unrelated",
        },
    )

    assert changes == []


def test_platform_overlay_is_idempotent():
    config = TrainerConfig.model_construct(monitors=MonitorsConfig(wandb=None), metrics_server=None)
    environ = {
        "PRL_JSON_LOGGING": "1",
        "PRL_OUTPUT_DIR": "/mnt/run-123",
        "PRL_FORCE_WANDB_MONITOR": "1",
        "PRL_RUN_ID": "run-123",
        "PRL_METRICS_PORT": "8123",
    }

    assert apply_platform_env(config, "trainer", environ)
    assert apply_platform_env(config, "trainer", environ) == []


@pytest.mark.parametrize(
    "name",
    [
        "PRL_FORCE_PRIME_MONITOR",
        "PRL_FORCE_WANDB_MONITOR",
        "PRL_JSON_LOGGING",
        "PRL_METRICS_PORT",
        "PRL_OUTPUT_DIR",
    ],
)
def test_platform_env_vars_are_protected_from_component_env(name: str):
    assert name in PROTECTED_ENV_VARS
    with pytest.raises(ValueError, match=name):
        reject_protected_env_vars({name: "user-value"})
