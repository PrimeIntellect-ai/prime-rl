import tomllib

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import INFERENCE_TOML, TEACHER_INFERENCE_TOML, write_subconfigs
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


def test_write_subconfigs_preserves_teacher_inference_single_node_ports(tmp_path):
    config = cli(
        RLConfig,
        args=[
            "@",
            "configs/ci/integration/rl/start.toml",
            "--inference.server.port",
            "9000",
            "--deployment.num_teacher_gpus",
            "1",
        ],
    )

    write_subconfigs(config, tmp_path)

    with open(tmp_path / TEACHER_INFERENCE_TOML, "rb") as f:
        teacher_inference_config = tomllib.load(f)

    validated_teacher_config = InferenceConfig.model_validate(teacher_inference_config)

    assert teacher_inference_config["server"]["port"] == 9001
    assert teacher_inference_config["deployment"] == {
        "type": "single_node",
        "gpus_per_node": 8,
        "router_port": 9001,
        "backend_port": 9101,
        "router_policy": "consistent_hash",
    }
    assert validated_teacher_config.server.port == 9001
