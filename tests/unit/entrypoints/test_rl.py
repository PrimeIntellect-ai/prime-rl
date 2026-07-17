import tomllib
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import INFERENCE_TOML, write_subconfigs


def test_write_subconfigs_preserves_dynamo_disaggregated_deployment(tmp_path: Path):
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {
                "backend": {"type": "dynamo"},
                "deployment": {
                    "type": "disaggregated",
                    "gpus_per_node": 1,
                    "num_prefill_replicas": 1,
                    "num_decode_replicas": 1,
                },
                "enable_expert_parallel": False,
            },
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 3,
                "num_train_gpus": 1,
                "num_infer_gpus": 2,
            },
            "slurm": {},
        }
    )

    write_subconfigs(config, tmp_path)

    with (tmp_path / INFERENCE_TOML).open("rb") as file:
        reloaded = InferenceConfig.model_validate(tomllib.load(file))

    assert reloaded.slurm is None
    assert reloaded.deployment.type == "disaggregated"
    assert reloaded.dynamo_worker_roles == ("prefill", "decode")
