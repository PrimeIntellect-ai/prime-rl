import json
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference import dynamo
from prime_rl.inference.dynamo import (
    build_engine_config,
    build_frontend_process,
    build_local_worker_specs,
    build_worker_environment,
    build_worker_process,
    write_role_engine_configs,
)


def disaggregated_config(**overrides) -> InferenceConfig:
    data = {
        "backend": {"type": "dynamo"},
        "weight_broadcast": {"type": "nccl"},
        "deployment": {
            "type": "disaggregated",
            "gpus_per_node": 1,
            "prefill_nodes_per_replica": 1,
            "decode_nodes_per_replica": 1,
            "num_prefill_replicas": 2,
            "num_decode_replicas": 2,
        },
    }
    data.update(overrides)
    return InferenceConfig.model_validate(data)


def test_role_engine_configs_share_nixl_and_only_prefill_publishes_events(tmp_path: Path):
    paths = write_role_engine_configs(disaggregated_config(), tmp_path)
    prefill = json.loads(paths["prefill"].read_text())
    decode = json.loads(paths["decode"].read_text())

    assert prefill["kv_transfer_config"] == decode["kv_transfer_config"]
    assert prefill["kv_transfer_config"]["kv_connector"] == "NixlConnector"
    assert prefill["kv_events_config"]["enable_kv_cache_events"] is True
    assert "kv_events_config" not in decode
    assert prefill["worker_extension_cls"].endswith("NCCLWeightUpdateWorker")
    assert decode["worker_extension_cls"] == prefill["worker_extension_cls"]


def test_role_overrides_are_isolated():
    config = disaggregated_config(
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 1,
            "prefill_nodes_per_replica": 1,
            "decode_nodes_per_replica": 1,
            "num_prefill_replicas": 2,
            "num_decode_replicas": 2,
            "prefill_vllm_overrides": {"max_num_batched_tokens": 8192},
            "decode_vllm_overrides": {"max_num_seqs": 64},
        }
    )

    prefill = build_engine_config(config, "prefill", kv_events_port=20080)
    decode = build_engine_config(config, "decode")

    assert prefill["max_num_batched_tokens"] == 8192
    assert "max_num_batched_tokens" not in decode
    assert decode["max_num_seqs"] == 64
    assert "max_num_seqs" not in prefill


@pytest.mark.parametrize("key", ["kv_transfer_config", "kv_events_config", "worker_extension_cls"])
def test_reserved_engine_override_is_rejected(key: str):
    config = disaggregated_config(vllm_extra={key: {}})
    with pytest.raises(ValueError, match="Dynamo-managed"):
        build_engine_config(config, "prefill", kv_events_port=20080)


def test_local_specs_allocate_four_workers_and_unique_ports(tmp_path: Path):
    specs = build_local_worker_specs(disaggregated_config(), tmp_path, gpu_ids=["4", "5", "6", "7"])

    assert [spec.role for spec in specs] == ["decode", "decode", "prefill", "prefill"]
    assert [spec.gpu_ids for spec in specs] == [("4",), ("5",), ("6",), ("7",)]
    assert len({spec.system_port for spec in specs}) == 4
    assert len({spec.process.environment()["VLLM_NIXL_SIDE_CHANNEL_PORT"] for spec in specs}) == 4
    prefill_configs = [
        json.loads(Path(spec.process.arguments[1]).read_text()) for spec in specs if spec.role == "prefill"
    ]
    assert [config["kv_events_config"]["endpoint"] for config in prefill_configs] == [
        "tcp://*:20080",
        "tcp://*:20081",
    ]
    assert all("--enable-rl" in spec.process.command() for spec in specs)


def test_wrapper_options_are_not_written_to_engine_json():
    engine = build_engine_config(disaggregated_config(), "prefill", kv_events_port=20080)
    assert "disaggregation_mode" not in engine
    assert "enable_rl" not in engine


def test_process_specs_own_canonical_commands_and_environment(tmp_path: Path):
    config = disaggregated_config(
        env_vars={"SHARED": "value"},
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 1,
            "prefill_nodes_per_replica": 1,
            "decode_nodes_per_replica": 1,
            "num_prefill_replicas": 1,
            "num_decode_replicas": 1,
            "prefill_env_vars": {"ROLE": "prefill"},
        },
    )

    frontend = build_frontend_process(config)
    prefill = build_worker_process(
        config,
        "prefill",
        tmp_path / "prefill.json",
        nixl_host="127.0.0.1",
        nixl_port=20100,
    )

    assert frontend.module == "dynamo.frontend"
    assert frontend.arguments[-1] == "--enable-engine-apis"
    assert frontend.environment()["DYN_ENABLE_RL"] == "1"
    assert prefill.module == "dynamo.vllm"
    assert prefill.arguments[-3:] == ("--disaggregation-mode", "prefill", "--enable-rl")
    assert prefill.environment()["ROLE"] == "prefill"
    assert prefill.environment()["VLLM_PLUGINS"] == "prime_rl"
    assert prefill.environment()["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "20100"


def test_worker_environment_applies_only_matching_role_overrides(tmp_path: Path):
    config = disaggregated_config(
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 1,
            "prefill_nodes_per_replica": 1,
            "decode_nodes_per_replica": 1,
            "num_prefill_replicas": 1,
            "num_decode_replicas": 1,
            "prefill_env_vars": {"ROLE_SETTING": "prefill"},
            "decode_env_vars": {"ROLE_SETTING": "decode"},
        }
    )
    decode, prefill = build_local_worker_specs(config, tmp_path, gpu_ids=["3", "7"])

    decode_env = build_worker_environment(decode, {"COMMON": "value"})
    prefill_env = build_worker_environment(prefill, {"COMMON": "value"})

    assert decode_env["ROLE_SETTING"] == "decode"
    assert prefill_env["ROLE_SETTING"] == "prefill"
    assert decode_env["CUDA_VISIBLE_DEVICES"] == "3"
    assert prefill_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert decode_env["VLLM_PLUGINS"] == "prime_rl"


def test_child_failure_tears_down_complete_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = disaggregated_config(output_dir=tmp_path)
    processes = []
    terminated = []

    class FakeProcess:
        def __init__(self, returncode):
            self.pid = 1000 + len(processes)
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def popen(*_args, **_kwargs):
        process = FakeProcess(7 if not processes else None)
        processes.append(process)
        return process

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(dynamo.subprocess, "Popen", popen)
    monkeypatch.setattr(dynamo.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(dynamo, "_terminate", terminated.append)

    with pytest.raises(SystemExit) as exc:
        dynamo.run_dynamo_local(config)

    assert exc.value.code == 7
    assert len(processes) == 5
    assert terminated == list(reversed(processes))
