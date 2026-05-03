from prime_rl.configs.inference import InferenceConfig
from prime_rl.entrypoints.inference import (
    build_single_node_backend_config,
    build_single_node_router_cmd,
    should_use_local_router,
    write_slurm_script,
)


def test_build_single_node_router_cmd_uses_internal_backend_port():
    config = InferenceConfig.model_validate(
        {
            "server": {"port": 9000},
            "parallel": {"dp": 4},
            "data_parallel_size_local": 2,
        }
    )

    cmd = build_single_node_router_cmd(config)

    assert cmd[0] == "vllm-router"
    assert cmd[cmd.index("--worker-urls") + 1] == "http://127.0.0.1:9100"
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("--intra-node-data-parallel-size") + 1] == "2"


def test_build_single_node_backend_config_moves_server_to_backend_port():
    config = InferenceConfig.model_validate({"server": {"port": 9000}})

    backend_config = build_single_node_backend_config(config)

    assert config.server.port == 9000
    assert backend_config.server.port == 9100


def test_build_single_node_router_cmd_uses_prime_log_level_env(monkeypatch):
    monkeypatch.setenv("PRIME_LOG_LEVEL", "warning")
    config = InferenceConfig.model_validate({"server": {"port": 9000}})

    cmd = build_single_node_router_cmd(config)

    assert cmd[cmd.index("--log-level") + 1] == "warning"


def test_should_use_local_router_for_single_node_default():
    config = InferenceConfig.model_validate({"server": {"port": 9000}})

    assert should_use_local_router(config) is True


def test_should_not_use_local_router_in_backend_only_mode(monkeypatch):
    monkeypatch.setenv("PRIME_RL_INFERENCE_BACKEND_ONLY", "1")
    config = InferenceConfig.model_validate({"server": {"port": 9000}})

    assert should_use_local_router(config) is False


def test_write_slurm_script_runs_multi_node_workers_in_backend_only_mode(tmp_path):
    config = InferenceConfig.model_validate({"deployment": {"type": "multi_node"}, "slurm": {}})
    script_path = tmp_path / "inference.sbatch"

    write_slurm_script(config, tmp_path / "inference.toml", script_path)

    script = script_path.read_text()
    assert script.count("PRIME_RL_INFERENCE_BACKEND_ONLY=1 uv run inference") == 1
