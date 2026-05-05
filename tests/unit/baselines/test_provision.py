from pathlib import Path

from prime_rl.baselines.config import BaselineConfig, LaunchConfig
from prime_rl.baselines.provision import DP_COORDINATOR_STARTUP_TIMEOUT, InferenceProvisioner


def test_external_provisioner_uses_configured_wait_timeout(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_wait_for_endpoint(base_url: str, api_key_var: str, timeout_s: float) -> None:
        seen["base_url"] = base_url
        seen["api_key_var"] = api_key_var
        seen["timeout_s"] = timeout_s

    monkeypatch.setattr("prime_rl.baselines.provision.wait_for_endpoint", fake_wait_for_endpoint)
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        base_url="http://127.0.0.1:8000",
        api_key_var="VLLM_API_KEY",
        launch=LaunchConfig(mode="external", wait_timeout_s=123.0),
    )

    with InferenceProvisioner(config) as endpoint:
        assert endpoint.base_url == "http://127.0.0.1:8000/v1"

    assert seen == {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_var": "VLLM_API_KEY",
        "timeout_s": 123.0,
    }


def test_local_provisioner_retries_vllm_dp_coordinator_startup_timeout(monkeypatch, tmp_path: Path):
    popen_calls = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return FakeProc()

    def fake_wait_for_local_endpoint(base_url, api_key_var, timeout_s, proc, log_path):
        if len(popen_calls) == 1:
            log_path.write_text(DP_COORDINATOR_STARTUP_TIMEOUT)
            raise RuntimeError("startup failed")

    monkeypatch.setattr("prime_rl.baselines.provision.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "prime_rl.baselines.provision._wait_for_local_endpoint",
        fake_wait_for_local_endpoint,
    )
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        api_key_var="VLLM_API_KEY",
        launch=LaunchConfig(mode="local", wait_timeout_s=123.0, server_start_retries=1),
    )

    with InferenceProvisioner(config) as endpoint:
        assert endpoint.base_url == "http://127.0.0.1:8000/v1"

    assert len(popen_calls) == 2


def test_local_provisioner_creates_vllm_rpc_base_path(monkeypatch, tmp_path: Path):
    rpc_base_path = tmp_path / "vllm-rpc"

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_wait_for_local_endpoint(*args, **kwargs):
        return None

    monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(rpc_base_path))
    monkeypatch.setattr("prime_rl.baselines.provision.subprocess.Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(
        "prime_rl.baselines.provision._wait_for_local_endpoint",
        fake_wait_for_local_endpoint,
    )
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path / "run",
        api_key_var="VLLM_API_KEY",
        launch=LaunchConfig(mode="local"),
    )

    with InferenceProvisioner(config):
        pass

    assert rpc_base_path.is_dir()


def test_srun_provisioner_pins_local_endpoint_to_current_slurm_node(monkeypatch, tmp_path: Path):
    popen_calls = []

    class FakeProc:
        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setenv("SLURMD_NODENAME", "nid000123")
    monkeypatch.setattr(
        "prime_rl.baselines.provision.subprocess.Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)) or FakeProc(),
    )
    monkeypatch.setattr("prime_rl.baselines.provision._wait_for_local_endpoint", lambda *args, **kwargs: None)
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        api_key_var="VLLM_API_KEY",
        launch=LaunchConfig(mode="srun", srun_job_id="4441280"),
    )

    with InferenceProvisioner(config):
        pass

    cmd = popen_calls[0][0][0]
    assert "--nodes=1" in cmd
    assert "--jobid=4441280" in cmd
    assert "--nodelist=nid000123" in cmd
