import json

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.utils.specdec import check_draft_swap


def test_draft_swap_requires_same_architecture_and_tensors(tmp_path):
    def draft(name, block_size=8, verifier="/ckpt/a", tensors=("fc.weight",)):
        directory = tmp_path / name
        directory.mkdir()
        config = {"block_size": block_size, "speculators_config": {"algorithm": "dspark", "verifier": verifier}}
        (directory / "config.json").write_text(json.dumps(config))
        save_file({tensor: torch.zeros(2, 3) for tensor in tensors}, directory / "model.safetensors")
        return directory

    served = draft("served")
    check_draft_swap(served, draft("refreshed", verifier="/ckpt/b"))
    check_draft_swap(draft("with-target-weights", tensors=("fc.weight", "lm_head.weight")), served)
    with pytest.raises(ValueError, match="block_size"):
        check_draft_swap(served, draft("wider", block_size=16))
    with pytest.raises(ValueError, match="confidence_head"):
        check_draft_swap(served, draft("extra", tensors=("fc.weight", "confidence_head.weight")))
