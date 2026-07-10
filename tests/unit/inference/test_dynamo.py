import json
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.entrypoints import inference as inference_entrypoint
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


@pytest.mark.parametrize(
    "key",
    [
        "data_parallel_rpc_port",
        "data_parallel_size",
        "data_parallel_size_local",
        "disaggregation_mode",
        "enable_prefix_caching",
        "enable_rl",
        "kv_transfer_config",
        "kv_events_config",
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "worker_extension_cls",
    ],
)
def test_reserved_engine_override_is_rejected(key: str):
    config = disaggregated_config(vllm_extra={key: {}})
    with pytest.raises(ValueError, match="Dynamo-managed"):
        build_engine_config(config, "prefill", kv_events_port=20080)


def test_reserved_role_engine_override_is_rejected():
    config = disaggregated_config(
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 1,
            "num_prefill_replicas": 1,
            "num_decode_replicas": 1,
            "prefill_vllm_overrides": {"tensor_parallel_size": 2},
        }
    )

    with pytest.raises(ValueError, match="prefill_vllm_overrides.*tensor_parallel_size"):
        build_engine_config(config, "prefill", kv_events_port=20080)


@pytest.mark.parametrize("key", sorted(dynamo._ENGINE_CONFIG_EXCLUDED))
def test_wrapper_only_global_engine_override_is_rejected(key: str):
    config = disaggregated_config(vllm_extra={key: "invalid"})

    with pytest.raises(ValueError, match=rf"vllm_extra.*{key}.*wrapper/server-only"):
        build_engine_config(config, "prefill", kv_events_port=20080)


@pytest.mark.parametrize("key", sorted(dynamo._ENGINE_CONFIG_EXCLUDED))
def test_wrapper_only_role_engine_override_is_rejected(key: str):
    config = disaggregated_config(
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 1,
            "num_prefill_replicas": 1,
            "num_decode_replicas": 1,
            "decode_vllm_overrides": {key: "invalid"},
        }
    )

    with pytest.raises(ValueError, match=rf"decode_vllm_overrides.*{key}.*wrapper/server-only"):
        build_engine_config(config, "decode")


def test_local_specs_allocate_four_workers_and_unique_ports(tmp_path: Path):
    specs = build_local_worker_specs(disaggregated_config(), tmp_path, gpu_ids=["4", "5", "6", "7"])

    assert [spec.role for spec in specs] == list(disaggregated_config().dynamo_worker_roles)
    assert [spec.gpu_ids for spec in specs] == [("4",), ("5",), ("6",), ("7",)]
    assert len({spec.system_port for spec in specs}) == 4
    assert len({spec.process.environment()["VLLM_NIXL_SIDE_CHANNEL_PORT"] for spec in specs}) == 4
    prefill_configs = [
        json.loads(Path(spec.process.arguments[1]).read_text()) for spec in specs if spec.role == "prefill"
    ]
    assert len({config["kv_events_config"]["endpoint"] for config in prefill_configs}) == 2
    assert all("--enable-rl" in spec.process.command() for spec in specs)


def test_local_multi_gpu_workers_allocate_globally_unique_coordinator_ports(tmp_path: Path):
    config = disaggregated_config(
        parallel={"tp": 1},
        deployment={
            "type": "disaggregated",
            "gpus_per_node": 2,
            "prefill_nodes_per_replica": 1,
            "decode_nodes_per_replica": 1,
            "num_prefill_replicas": 1,
            "num_decode_replicas": 1,
        },
    )

    specs = build_local_worker_specs(config, tmp_path, gpu_ids=["0", "1", "2", "3"])
    engine_configs = [json.loads(Path(spec.process.arguments[1]).read_text()) for spec in specs]
    allocated_ports = [
        *(spec.system_port for spec in specs),
        *(int(spec.process.environment()["VLLM_NIXL_SIDE_CHANNEL_PORT"]) for spec in specs),
        *(engine["data_parallel_rpc_port"] for engine in engine_configs),
        *(
            int(engine["kv_events_config"]["endpoint"].rsplit(":", 1)[1])
            for engine in engine_configs
            if "kv_events_config" in engine
        ),
    ]

    assert len({engine["data_parallel_rpc_port"] for engine in engine_configs}) == len(specs)
    assert len(allocated_ports) == len(set(allocated_ports))


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


def test_process_specs_preserve_custom_chat_template_and_parsers(tmp_path: Path):
    source = tmp_path / "source-template.jinja"
    source.write_text("{{ messages | length }}")
    config = disaggregated_config(
        model={
            "chat_template": str(source),
            "tool_call_parser": "hermes",
            "reasoning_parser": "qwen3",
        }
    )

    frontend = build_frontend_process(config, output_dir=tmp_path / "generated")
    worker = build_worker_process(
        config,
        "decode",
        tmp_path / "decode.json",
        nixl_host=None,
        nixl_port=20100,
    )

    template_path = Path(frontend.arguments[frontend.arguments.index("--chat-template") + 1])
    assert template_path == tmp_path / "generated" / "chat-template.jinja"
    assert template_path.read_text() == source.read_text()
    assert frontend.arguments[-4:-2] == ("--dyn-chat-processor", "vllm")
    assert worker.arguments[-4:] == (
        "--dyn-tool-call-parser",
        "hermes",
        "--dyn-reasoning-parser",
        "qwen3",
    )


@pytest.mark.parametrize(
    ("role", "component"),
    [("prefill", "prefill"), ("decode", "backend"), ("agg", "backend")],
)
def test_worker_process_uses_deterministic_role_endpoint(tmp_path: Path, role: str, component: str):
    process = build_worker_process(
        disaggregated_config(env_vars={"DYN_NAMESPACE": "prime-test"}),
        role,
        tmp_path / f"{role}.json",
        nixl_host=None,
        nixl_port=20100,
    )

    endpoint = f"dyn://prime-test.{component}.generate"
    assert process.environment()["DYN_NAMESPACE"] == "prime-test"
    assert process.environment()["DYN_COMPONENT"] == component
    assert process.environment()["DYN_ENDPOINT"] == endpoint
    assert process.arguments[2:4] == ("--endpoint", endpoint)


def test_inline_chat_template_is_materialized_verbatim(tmp_path: Path):
    config = disaggregated_config(model={"chat_template": "{{ messages }}"})

    frontend = build_frontend_process(config, output_dir=tmp_path)

    template_path = Path(frontend.arguments[-1])
    assert template_path.read_text() == "{{ messages }}"


def test_frontend_runtime_chat_template_path_does_not_materialize_host_file(tmp_path: Path):
    config = disaggregated_config(model={"chat_template": "{{ messages }}"})
    output_dir = tmp_path / "render-host"
    runtime_path = Path("/etc/prime-rl/dynamo/chat-template.jinja")

    frontend = build_frontend_process(
        config,
        output_dir=output_dir,
        runtime_chat_template_path=runtime_path,
    )

    assert frontend.arguments[-1] == str(runtime_path)
    assert not output_dir.exists()


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
    prefill, decode = build_local_worker_specs(config, tmp_path, gpu_ids=["3", "7"])

    decode_env = build_worker_environment(decode, {"COMMON": "value"})
    prefill_env = build_worker_environment(prefill, {"COMMON": "value"})

    assert decode_env["ROLE_SETTING"] == "decode"
    assert prefill_env["ROLE_SETTING"] == "prefill"
    assert prefill_env["CUDA_VISIBLE_DEVICES"] == "3"
    assert decode_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert decode_env["VLLM_PLUGINS"] == "prime_rl"
    assert decode_env["DYN_COMPONENT"] == "backend"
    assert prefill_env["DYN_COMPONENT"] == "prefill"


def test_aggregated_worker_uses_canonical_component_name(tmp_path: Path):
    config = InferenceConfig.model_validate({"backend": {"type": "dynamo"}})
    spec = build_local_worker_specs(config, tmp_path, gpu_ids=["0"])[0]

    assert build_worker_environment(spec, {})["DYN_COMPONENT"] == "backend"


def test_dynamo_dry_run_uses_symbolic_gpu_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = disaggregated_config(output_dir=tmp_path, dry_run=True)
    captured_gpu_ids = []
    original_build_specs = dynamo.build_local_worker_specs

    class FakeLogger:
        def info(self, _message):
            pass

        def success(self, _message):
            pass

    def build_specs(config, output_dir=None, gpu_ids=None, namespace=None):
        captured_gpu_ids.extend(gpu_ids or [])
        return original_build_specs(config, output_dir=output_dir, gpu_ids=gpu_ids, namespace=namespace)

    monkeypatch.setattr(dynamo, "_visible_gpu_ids", lambda: pytest.fail("dry-run queried physical GPUs"))
    monkeypatch.setattr(dynamo, "build_local_worker_specs", build_specs)
    monkeypatch.setattr(inference_entrypoint, "setup_logger", lambda *_args, **_kwargs: FakeLogger())

    inference_entrypoint.inference_local(config)

    assert captured_gpu_ids == ["<gpu:0>", "<gpu:1>", "<gpu:2>", "<gpu:3>"]


@pytest.mark.parametrize(("child_code", "supervisor_code"), [(7, 7), (0, 1)])
def test_child_exit_tears_down_complete_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_code: int,
    supervisor_code: int,
):
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
        process = FakeProcess(child_code if not processes else None)
        processes.append(process)
        return process

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(dynamo.subprocess, "Popen", popen)
    monkeypatch.setattr(dynamo.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(dynamo, "_terminate", terminated.append)

    with pytest.raises(SystemExit) as exc:
        dynamo.run_dynamo_local(config)

    assert exc.value.code == supervisor_code
    assert len(processes) == 5
    assert terminated == list(reversed(processes))
