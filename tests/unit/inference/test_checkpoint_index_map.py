import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.worker.weight_transfer import CheckpointIndexMap


class Toy(nn.Module):
    """An engine model that keeps a norm in fp32 (cast on load from the bf16 checkpoint)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16))
        self.norm = nn.Parameter(torch.zeros(8, dtype=torch.float32))

    def load_weights(self, weights):
        for name, tensor in weights:
            getattr(self, name).data.copy_(tensor)


def test_cast_tensors_load_by_value_and_reject_deltas():
    model = Toy()
    index_map = CheckpointIndexMap(model, {"weight": ([4, 8], "bfloat16"), "norm": ([8], "bfloat16")})
    assert index_map.row_ranges() == {"weight": (0, 4), "norm": (0, 1)}

    norm, weight = torch.randn(1, 8, dtype=torch.bfloat16), torch.randn(4, 8, dtype=torch.bfloat16)
    index_map.load_rows("norm", norm, 0)
    index_map.load_rows("weight", weight, 0)
    assert torch.equal(model.norm.data, norm.view(-1).float())
    assert torch.equal(model.weight.data, weight)

    with pytest.raises(NotImplementedError):
        index_map.apply_delta([("norm", 1)], torch.tensor([0]), torch.tensor([1], dtype=torch.int16))
