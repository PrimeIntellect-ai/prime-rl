import signal

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference import dynamo_launcher
from prime_rl.inference.dynamo_launcher import build_dynamo_process_specs


def managed_config(*, transport="nccl", server_port=8000, env_vars=None, **vllm) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "backend": "dynamo",
            "server": {"host": "127.0.0.1", "port": server_port},
            "router": None,
            "weight_broadcast": {"type": transport},
            "env_vars": env_vars or {},
            "vllm": {"model": "Qwen/Qwen3-0.6B", **vllm},
        }
    )


def test_managed_dynamo_process_specs_use_frontend_and_rl_worker():
    frontend, worker = build_dynamo_process_specs(managed_config(max_model_len=2048), executable="/python")

    assert frontend.name == "frontend"
    assert frontend.command == ("/python", "-m", "dynamo.frontend")
    assert frontend.environment["DYN_HTTP_HOST"] == "127.0.0.1"
    assert frontend.environment["DYN_HTTP_PORT"] == "8000"
    assert frontend.environment["DYN_RL_PORT"] == "8001"
    assert frontend.environment["DYN_ENABLE_RL"] == "true"
    assert frontend.environment["DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE"] == "1"
    assert frontend.environment["CUDA_VISIBLE_DEVICES"] == ""

    assert worker.name == "worker"
    assert worker.command[:4] == ("/python", "-m", "dynamo.vllm", "--enable-rl")
    assert worker.command[4:6] == ("--model", "Qwen/Qwen3-0.6B")
    max_model_len_index = worker.command.index("--max-model-len")
    assert worker.command[max_model_len_index : max_model_len_index + 2] == ("--max-model-len", "2048")
    assert worker.command[-2:] == (
        "--worker-extension-cls",
        "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    )
    additional_config_index = worker.command.index("--additional-config")
    assert worker.command[additional_config_index : additional_config_index + 2] == (
        "--additional-config",
        '{"fp32_lm_head":true}',
    )
    assert "moe_router_dtype" in worker.command[worker.command.index("--hf-overrides") + 1]
    assert worker.environment["DYN_SYSTEM_PORT"] == "8081"
    assert worker.environment["VLLM_PLUGINS"] == "prime_rl"
    assert "CUDA_VISIBLE_DEVICES" not in worker.environment


def test_managed_dynamo_worker_serializes_explicit_vllm_values():
    _, worker = build_dynamo_process_specs(
        managed_config(enforce_eager=True, gpu_memory_utilization=0.75, max_num_seqs=16),
        executable="/python",
    )

    assert "--enforce-eager" in worker.command
    index = worker.command.index("--gpu-memory-utilization")
    assert worker.command[index : index + 2] == ("--gpu-memory-utilization", "0.75")
    index = worker.command.index("--max-num-seqs")
    assert worker.command[index : index + 2] == ("--max-num-seqs", "16")


def test_managed_dynamo_rejects_credential_worker_arguments():
    with pytest.raises(ValueError, match="must be provided through the environment"):
        build_dynamo_process_specs(managed_config(hf_token="test-value"))


def test_managed_dynamo_rejects_unsupported_transport_and_port_collision():
    with pytest.raises(ValueError, match="NCCL and NIXL"):
        build_dynamo_process_specs(managed_config(transport="filesystem"))

    with pytest.raises(ValueError, match="distinct"):
        build_dynamo_process_specs(managed_config(server_port=8080, env_vars={"DYN_SYSTEM_PORT": "8081"}))


def test_managed_dynamo_child_failure_stops_both_processes(monkeypatch):
    processes = []
    terminated = []
    environments = []

    class FakeProcess:
        def __init__(self, returncode):
            self.pid = 1000 + len(processes)
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def popen(*_args, **kwargs):
        process = FakeProcess(None if not processes else 7)
        processes.append(process)
        environments.append(kwargs["env"])
        return process

    monkeypatch.setenv("DYN_DISCOVERY_BACKEND", "etcd")
    monkeypatch.setenv("DYN_EVENT_PLANE", "nats")
    monkeypatch.setenv("DYN_NAMESPACE", "shared")
    monkeypatch.setenv("DYN_FILE_KV", "/shared")
    monkeypatch.setattr(dynamo_launcher.subprocess, "Popen", popen)
    monkeypatch.setenv("DYN_REQUEST_PLANE", "nats")
    monkeypatch.setenv("HF_TOKEN", "not-for-frontend")
    monkeypatch.setenv("KUBECONFIG", "not-for-frontend")
    monkeypatch.setattr(dynamo_launcher, "_terminate", terminated.append)

    with pytest.raises(RuntimeError, match="Dynamo worker exited with code 7"):
        dynamo_launcher.run_dynamo_local(managed_config())
    assert len(processes) == 2
    assert terminated == list(reversed(processes))

    assert {environment["DYN_DISCOVERY_BACKEND"] for environment in environments} == {"file"}
    assert {environment["DYN_EVENT_PLANE"] for environment in environments} == {"zmq"}
    assert {environment["DYN_NAMESPACE"] for environment in environments} != {"shared"}
    assert {environment["DYN_FILE_KV"] for environment in environments} != {"/shared"}
    assert len({environment["DYN_FILE_KV"] for environment in environments}) == 1
    assert {environment["DYN_REQUEST_PLANE"] for environment in environments} == {"tcp"}
    assert "HF_TOKEN" not in environments[0]
    assert "KUBECONFIG" not in environments[0]
    assert environments[1]["HF_TOKEN"] == "not-for-frontend"
    assert environments[1]["KUBECONFIG"] == "not-for-frontend"


def test_managed_dynamo_uses_explicit_discovery_port_and_composes_plugins():
    frontend, worker = build_dynamo_process_specs(
        managed_config(env_vars={"DYN_RL_PORT": "9000", "VLLM_PLUGINS": "custom,prime_rl"})
    )

    assert frontend.environment["DYN_RL_PORT"] == "9000"
    assert worker.environment["VLLM_PLUGINS"] == "custom,prime_rl"


def test_terminate_signals_process_group_after_leader_exit(monkeypatch):
    signals = []

    class ExitedProcess:
        pid = 1234

        def wait(self, timeout=None):
            assert timeout == 15

    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(dynamo_launcher.os, "killpg", killpg)
    dynamo_launcher._terminate(ExitedProcess())

    assert signals == [(1234, signal.SIGTERM), (1234, 0)]


def test_managed_dynamo_worker_omits_none_vllm_values():
    _, worker = build_dynamo_process_specs(
        managed_config(enable_prefix_caching=None, quantization=None),
        executable="/python",
    )

    assert "--enable-prefix-caching" not in worker.command
    assert "--quantization" not in worker.command
