"""TTT launch hardening: RLConfig.validate_ttt and the launcher-managed service wiring.
TOML parse/launchability of the shipped configs (incl. arm overlay composition) is covered
by tests/unit/test_configs.py."""

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from prime_rl.configs.rl import RLConfig  # noqa: E402

CONFIGS = Path(__file__).parents[3] / "configs" / "ttt"


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_scaleswe_lora_sizing():
    base = load(CONFIGS / "scaleswe" / "base.toml")["inference"]
    service = load(CONFIGS / "scaleswe" / "ttt_service.toml")
    # vLLM requires max_cpu_loras >= max_loras (its default of 100 would abort the launch).
    assert base["max_cpu_loras"] >= base["max_loras"]
    # The engine can only serve adapters whose rank fits max_lora_rank.
    assert base["max_lora_rank"] >= service["lora"]["rank"]


# --- RLConfig.validate_ttt launch hardening ---


def rl_payload(ttt: dict | None, **inference_overrides) -> dict:
    """Minimal RLConfig payload with one (legacy-id) train env carrying a ``ttt`` block."""
    env: dict = {"id": "dummy-env"}
    if ttt is not None:
        env["ttt"] = ttt
    inference = {"enable_lora": True, "max_loras": 16, "max_cpu_loras": 16, **inference_overrides}
    return {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "trainer": {},
        "orchestrator": {"batch_size": 16, "group_size": 1, "train": {"env": [env]}},
        "inference": inference,
    }


def test_disabled_ttt_imposes_no_constraints():
    """enabled=false is the master switch: no LoRA-serving requirement, no ttt_replay."""
    config = RLConfig.model_validate(
        rl_payload({"base_url": "http://localhost:8092", "enabled": False}, enable_lora=False)
    )
    assert config.trainer.ttt_replay is False


def test_active_ttt_sets_ttt_replay():
    config = RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}))
    assert config.trainer.ttt_replay is True


def test_recycle_and_meta_lessons_mutually_exclusive():
    """A4 (recycle_to_policy) and A5 (meta_lessons) must stay disentangled."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        RLConfig.model_validate(
            rl_payload({"base_url": "http://localhost:8092", "qa": {"recycle_to_policy": True, "meta_lessons": True}})
        )




def test_nonpositive_qa_temperature_rejected():
    with pytest.raises(ValueError, match="qa.temperature must be > 0"):
        RLConfig.model_validate(
            rl_payload({"base_url": "http://localhost:8092", "qa": {"temperature": 0}})
        )

def test_inflight_exceeding_max_loras_rejected():
    """Every in-flight rollout holds a vLLM adapter slot — inflight must fit max_loras."""
    with pytest.raises(ValueError, match="max_loras"):
        RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}, max_loras=8))


def test_inflight_equal_to_max_loras_ok():
    # batch_size=16 resolves max_inflight_rollouts=16 == max_loras.
    RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}, max_loras=16))


def eval_only_ttt_payload(**inference_overrides) -> dict:
    """RLConfig payload where only the EVAL env carries a ``ttt`` block."""
    payload = rl_payload(None, **inference_overrides)
    payload["orchestrator"]["eval"] = {"env": [{"id": "dummy-eval-env", "ttt": {"base_url": "http://localhost:8092"}}]}
    return payload


def test_eval_env_ttt_requires_enable_lora():
    """Eval envs run the identical TTT inference regime, so they impose the same requirement."""
    with pytest.raises(ValueError, match="enable_lora"):
        RLConfig.model_validate(eval_only_ttt_payload(enable_lora=False))


def test_eval_only_ttt_does_not_set_ttt_replay():
    """Eval adapters are dismissed per rollout, never replayed — no trainer replay hooks."""
    config = RLConfig.model_validate(eval_only_ttt_payload())
    assert config.trainer.ttt_replay is False


# --- Launcher-managed TTT service (RLConfig.ttt + deployment.num_ttt_nodes) ---


def rl_ttt_service_payload(**overrides) -> dict:
    payload = rl_payload({"base_url": "auto"}, max_lora_rank=16, max_loras=64, max_cpu_loras=64)
    payload["deployment"] = {
        "type": "multi_node",
        "num_train_nodes": 1,
        "num_infer_nodes": 1,
        "num_ttt_nodes": 1,
    }
    payload["slurm"] = {"job_name": "t"}  # multi_node requires SLURM
    payload["ttt"] = {"engine": {"type": "fsdp", "max_slots": 64}, "lora": {"rank": 16}}
    for key, value in overrides.items():
        payload[key] = value
    return payload




def test_managed_eval_ttt_rejects_policy_lora():
    payload = rl_ttt_service_payload()
    payload["orchestrator"]["train"]["env"] = [{"id": "dummy-env"}]
    payload["orchestrator"]["eval"] = {
        "env": [{"id": "dummy-eval-env", "ttt": {"base_url": "auto"}}]
    }
    payload["trainer"]["model"] = {"lora": {}}
    with pytest.raises(ValueError, match="policy LoRA broadcasts cannot update"):
        RLConfig.model_validate(payload)

def test_ttt_service_autowires_output_dir_and_model():
    config = RLConfig.model_validate(rl_ttt_service_payload())
    assert config.ttt.output_dir == config.output_dir
    assert config.ttt.model.name == "Qwen/Qwen3-0.6B"
    assert config.ttt.engine.to_model_config(config.ttt.lora).name == "Qwen/Qwen3-0.6B"


def test_ttt_service_autowires_component_level_policy_model():
    """Without a shared top-level [model], the resolved trainer model stays authoritative."""
    payload = rl_ttt_service_payload()
    del payload["model"]
    policy_model = "zai-org/GLM-4.5-Air"
    payload["trainer"]["model"] = {"name": policy_model}
    payload["orchestrator"]["model"] = {"name": policy_model}
    payload["inference"]["model"] = {"name": policy_model}
    config = RLConfig.model_validate(payload)
    assert config.ttt.model.name == policy_model
    assert config.ttt.engine.to_model_config(config.ttt.lora).name == policy_model


def test_managed_service_contract_ignores_external_ttt_envs():
    payload = rl_ttt_service_payload()
    managed = payload["orchestrator"]["train"]["env"][0]
    managed["ttt"]["adapter_prefix"] = "managed"
    external = {
        "id": "external-env",
        "ttt": {"base_url": "http://external:9000", "adapter_prefix": "external"},
    }
    payload["orchestrator"]["train"]["env"].append(external)
    config = RLConfig.model_validate(payload)
    assert config.ttt.adapter_prefix == "managed"

    payload["orchestrator"]["train"]["env"] = [external]
    with pytest.raises(ValueError, match='no active train/eval env uses ttt.base_url = "auto"'):
        RLConfig.model_validate(payload)


def _drop_service_config(payload):
    del payload["ttt"]


def _zero_ttt_nodes(payload):
    payload["deployment"]["num_ttt_nodes"] = 0


def _oversized_rank(payload):
    payload["ttt"]["lora"]["rank"] = 32  # > max_lora_rank 16


def _inflight_exceeds_slots(payload):
    payload["orchestrator"]["max_inflight_rollouts"] = 65
    payload["inference"]["max_loras"] = 128


def _no_ttt_envs(payload):
    payload["orchestrator"]["train"]["env"] = [{"id": "dummy-env"}]  # no ttt block


def _nodes_without_service_or_envs(payload):
    del payload["ttt"]
    payload["orchestrator"]["train"]["env"] = [{"id": "dummy-env"}]


def _single_node(payload):
    del payload["deployment"]
    del payload["slurm"]


def _no_launched_inference(payload):
    payload["deployment"]["num_infer_nodes"] = 0
    payload["inference"] = None
    payload["trainer"] = {"data": {"fake": {}}}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(_drop_service_config, r"no \[ttt\] service config", id="nodes-without-service-config"),
        pytest.param(_zero_ttt_nodes, "never start it", id="service-without-nodes"),
        pytest.param(_oversized_rank, "max_lora_rank", id="rank-exceeds-max-lora-rank"),
        pytest.param(_inflight_exceeds_slots, "max_slots", id="slots-below-inflight"),
        # [ttt]/num_ttt_nodes with no env carrying an active ttt block is dead config.
        pytest.param(_no_ttt_envs, "no train/eval env has ttt enabled", id="service-without-ttt-envs"),
        # num_ttt_nodes > 0 without [ttt] fails even when no env has TTT (hoisted check).
        pytest.param(_nodes_without_service_or_envs, r"no \[ttt\] service config", id="nodes-without-ttt-envs"),
        # Non-multi_node deployments never start the service — [ttt] is meaningless there.
        pytest.param(_single_node, "never start it", id="service-on-single-node"),
        # The multi-node template only wires managed TTT to launched inference replicas.
        pytest.param(_no_launched_inference, "requires a launched inference deployment", id="no-launched-inference"),
    ],
)
def test_ttt_service_misconfigurations_rejected(mutate, match):
    payload = rl_ttt_service_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        RLConfig.model_validate(payload)


def test_orchestrator_ttt_base_url_fans_out():
    from prime_rl.configs.orchestrator import OrchestratorConfig

    config = OrchestratorConfig.model_validate(
        {
            "batch_size": 8,
            "group_size": 2,
            "train": {"env": [{"id": "dummy-env", "ttt": {"base_url": "auto"}}, {"id": "dummy-env2"}]},
            "ttt_base_url": "http://ttt-node:8092",
        }
    )
    assert config.train.env[0].ttt.base_url == "http://ttt-node:8092"
    assert getattr(config.train.env[1], "ttt", None) is None  # non-TTT env untouched


def test_orchestrator_ttt_base_url_fanout_is_placeholder_only():
    """An explicit per-env URL must survive the fan-out; only "auto" is overwritten."""
    from prime_rl.configs.orchestrator import OrchestratorConfig

    config = OrchestratorConfig.model_validate(
        {
            "batch_size": 8,
            "group_size": 2,
            "train": {
                "env": [
                    {"id": "dummy-env", "ttt": {"base_url": "auto"}},
                    {"id": "dummy-env2", "ttt": {"base_url": "http://external:9000"}},
                ]
            },
            "ttt_base_url": "http://ttt-node:8092",
        }
    )
    assert config.train.env[0].ttt.base_url == "http://ttt-node:8092"
    assert config.train.env[1].ttt.base_url == "http://external:9000"


def test_orchestrator_leftover_auto_rejected_at_startup():
    """A leftover "auto" without the launcher-injected URL must fail fast at orchestrator
    startup, not POST to 'auto'; parse-time it stays legal (SLURM injects the URL later)."""
    from prime_rl.configs.orchestrator import OrchestratorConfig

    config = OrchestratorConfig.model_validate(
        {
            "batch_size": 8,
            "group_size": 2,
            "train": {"env": [{"id": "dummy-env", "ttt": {"base_url": "auto"}}]},
        }
    )
    with pytest.raises(ValueError, match='"auto" requires the launcher-managed'):
        config.check_ttt_base_urls_resolved()
    # With the URL injected the same config passes the startup check.
    config.ttt_base_url = "http://ttt-node:8092"
    config.apply_ttt_base_url()
    config.check_ttt_base_urls_resolved()


# --- Policy weight-broadcast receiver wiring (TTT service follows the policy) ---


def test_ttt_service_autowires_nccl_weight_broadcast_receiver():
    """multi_node + nccl: trainer group grows by the TTT GPUs and the service receiver
    occupies global receiver ranks [inference_world_size .. +num_ttt_nodes*gpus)."""
    config = RLConfig.model_validate(rl_ttt_service_payload())
    assert config.trainer.weight_broadcast.type == "nccl"
    assert config.trainer.weight_broadcast.ttt_world_size == 8  # 1 ttt node * 8 gpus
    assert config.orchestrator.weight_broadcast.ttt_world_size == 8
    inference_world_size = config.trainer.weight_broadcast.inference_world_size
    wb = config.ttt.weight_broadcast
    assert wb is not None and wb.type == "nccl"
    assert wb.rank_offset == inference_world_size
    assert wb.world_size == inference_world_size + 8
    assert wb.port == config.weight_broadcast.port
    assert wb.timeout == config.weight_broadcast.timeout


def test_ttt_service_receiver_mismatch_rejected():
    payload = rl_ttt_service_payload()
    payload["ttt"]["weight_broadcast"] = {"type": "nccl", "rank_offset": 3, "world_size": 5}
    with pytest.raises(ValueError, match="conflicts with the run's NCCL broadcast wiring"):
        RLConfig.model_validate(payload)


def test_ttt_service_follows_filesystem_broadcast():
    payload = rl_ttt_service_payload()
    payload["weight_broadcast"] = {"type": "filesystem"}
    config = RLConfig.model_validate(payload)
    assert config.ttt.weight_broadcast is not None
    assert config.ttt.weight_broadcast.type == "filesystem"
    assert config.trainer.weight_broadcast.type == "filesystem"


def test_ttt_quantized_transfer_rejected_with_managed_service():
    payload = rl_ttt_service_payload()
    payload["weight_broadcast"] = {"type": "nccl", "quantize_in_weight_transfer": True}
    payload["trainer"]["model"] = {"impl": "custom"}
    with pytest.raises(ValueError, match="quantiz"):
        RLConfig.model_validate(payload)


def test_standalone_service_defaults_to_no_weight_broadcast():
    from prime_rl.configs.ttt import TTTServiceConfig

    assert TTTServiceConfig().weight_broadcast is None
    with pytest.raises(ValueError, match="fsdp"):
        TTTServiceConfig(weight_broadcast={"type": "nccl"})  # receive path is v2-only


def test_trainer_broadcast_without_ttt_keeps_zero_ttt_world_size():
    config = RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}))
    if config.trainer.weight_broadcast.type == "nccl":
        assert config.trainer.weight_broadcast.ttt_world_size == 0


def test_scaleswe_arm_overlays_get_receiver_autowiring():
    """The shipped scaleswe TTT arms still validate and pick up the auto-wiring."""
    import copy

    def deep_merge(base: dict, overlay: dict) -> dict:
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    payload = deep_merge(load(CONFIGS / "scaleswe" / "base.toml"), load(CONFIGS / "scaleswe" / "ttt_common.toml"))
    payload = deep_merge(payload, load(CONFIGS / "scaleswe" / "arm_a2_ttt.toml"))
    config = RLConfig.model_validate(payload)
    assert config.trainer.weight_broadcast.type == "nccl"
    # 4 replicas x 1 node x 8 GPUs inference + 1 TTT node x 8 GPUs + trainer rank 0 = 41.
    assert config.trainer.weight_broadcast.inference_world_size == 32
    assert config.trainer.weight_broadcast.ttt_world_size == 8
    assert config.ttt.weight_broadcast.rank_offset == 32
    assert config.ttt.weight_broadcast.world_size == 40
