import json

import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from prime_rl.utils.sparse_weights import apply_sparse_delta, load_safetensors, save_safetensors, write_sparse_manifest


def test_apply_sparse_delta_updates_sharded_checkpoint(tmp_path):
    full_dir = tmp_path / "full"
    delta_dir = tmp_path / "delta"
    full_dir.mkdir()
    delta_dir.mkdir()

    save_safetensors(
        {
            "model.layers.0.weight": torch.tensor([1, 2, 3], dtype=torch.bfloat16),
            "model.layers.0.bias": torch.tensor([4, 5], dtype=torch.bfloat16),
        },
        full_dir / "model-00001-of-00002.safetensors",
    )
    save_safetensors(
        {"model.layers.1.weight": torch.tensor([6, 7, 8], dtype=torch.bfloat16)},
        full_dir / "model-00002-of-00002.safetensors",
    )
    with open(full_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.bias": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors",
                },
            },
            f,
        )

    save_safetensors(
        {
            "model.layers.0.weight.indices": torch.tensor([1], dtype=torch.int64),
            "model.layers.0.weight.values": torch.tensor([20], dtype=torch.bfloat16),
            "model.layers.1.weight.indices": torch.tensor([0, 2], dtype=torch.int64),
            "model.layers.1.weight.values": torch.tensor([60, 80], dtype=torch.bfloat16),
        },
        delta_dir / "delta-00001-of-00001.safetensors",
    )
    write_sparse_manifest(
        delta_dir,
        {
            "type": "delta",
            "base_step": 1,
            "step": 2,
            "patch_files": [
                {
                    "file": "delta-00001-of-00001.safetensors",
                    "tensors": [
                        {
                            "name": "model.layers.0.weight",
                            "indices_key": "model.layers.0.weight.indices",
                            "values_key": "model.layers.0.weight.values",
                        },
                        {
                            "name": "model.layers.1.weight",
                            "indices_key": "model.layers.1.weight.indices",
                            "values_key": "model.layers.1.weight.values",
                        },
                    ],
                }
            ],
        },
    )

    apply_sparse_delta(delta_dir, full_dir)

    first_shard = load_safetensors(full_dir / "model-00001-of-00002.safetensors")
    second_shard = load_safetensors(full_dir / "model-00002-of-00002.safetensors")
    assert first_shard["model.layers.0.weight"].tolist() == [1, 20, 3]
    assert first_shard["model.layers.0.bias"].tolist() == [4, 5]
    assert second_shard["model.layers.1.weight"].tolist() == [60, 7, 80]
