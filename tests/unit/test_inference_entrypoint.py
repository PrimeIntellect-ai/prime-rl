from prime_rl.configs.inference import InferenceConfig
from prime_rl.entrypoints.inference import build_single_node_backend_config, build_single_node_router_cmd


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
