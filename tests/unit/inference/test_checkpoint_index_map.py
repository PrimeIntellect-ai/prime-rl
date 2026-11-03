import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.worker.weight_transfer import CheckpointIndexMap
from prime_rl.utils.sparse_weights import as_bits, bits_checksum


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


def test_running_checksums_follow_deltas():
    model = Toy()
    index_map = CheckpointIndexMap(model, {"weight": ([4, 8], "bfloat16"), "norm": ([8], "bfloat16")})
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    index_map.load_rows("weight", weight, 0)
    names, totals, held = index_map.checksums()
    assert names == ["weight"] and held.tolist() == [32]
    assert totals.tolist() == [bits_checksum(as_bits(weight))]

    # Raw-bit differences, including ones that wrap around the int16 range
    indices = torch.tensor([0, 5, 6, 31])
    delta = torch.tensor([1, -3, 32767, -32768], dtype=torch.int16)
    index_map.apply_delta([("weight", 4)], indices, delta)
    new = as_bits(weight).clone().view(-1)
    new[indices] += delta
    _, running, _ = index_map.checksums()
    assert running.tolist() == [bits_checksum(new)]
    assert running.tolist() == index_map.checksums(full=True)[1].tolist()
