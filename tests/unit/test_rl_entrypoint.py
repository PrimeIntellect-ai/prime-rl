import tomllib

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import INFERENCE_TOML, write_subconfigs
from prime_rl.utils.config import cli


def test_write_subconfigs_preserves_single_node_inference_deployment(tmp_path):
    config = cli(
        RLConfig,
        args=[
            "@",
            "configs/ci/integration/rl/start.toml",
            "--inference.deployment.router_port",
            "9000",
            "--inference.deployment.backend_port",
            "9105",
            "--inference.deployment.router_policy",
            "round_robin",
        ],
    )

    write_subconfigs(config, tmp_path)

    with open(tmp_path / INFERENCE_TOML, "rb") as f:
        inference_config = tomllib.load(f)

    assert inference_config["server"]["port"] == 9000
    assert inference_config["deployment"] == {
        "type": "single_node",
        "gpus_per_node": 8,
        "router_port": 9000,
        "backend_port": 9105,
        "router_policy": "round_robin",
    }
