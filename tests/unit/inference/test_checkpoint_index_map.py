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


class Sliced(nn.Module):
    """Checkpoint tensors written as row halves (like TP slices), through a column-strided view, and as fp32."""

    def __init__(self, device):
        super().__init__()
        self.top = nn.Parameter(torch.zeros(4, 16, dtype=torch.bfloat16, device=device))
        self.bottom = nn.Parameter(torch.zeros(4, 16, dtype=torch.bfloat16, device=device))
        self.packed = nn.Parameter(torch.zeros(6, 10, dtype=torch.bfloat16, device=device))
        self.scale = nn.Parameter(torch.zeros(32, dtype=torch.float32, device=device))

    def load_weights(self, weights):
        for name, tensor in weights:
            if name == "w":
                self.top.data.copy_(tensor[:4])
                self.bottom.data.copy_(tensor[4:])
            elif name == "v":
                self.packed.data[:, 2:7].copy_(tensor)
            else:
                self.scale.data.copy_(tensor)


@pytest.mark.gpu
def test_fused_apply_matches_torch_apply():
    tensors = {"w": ([8, 16], "bfloat16"), "v": ([6, 5], "bfloat16"), "s": ([32], "float32")}
    checkpoint = {
        "w": torch.randn(8, 16, dtype=torch.bfloat16),
        "v": torch.randn(6, 5, dtype=torch.bfloat16),
        "s": torch.randn(32, dtype=torch.float32),
    }
    maps = {}
    for device in ("cpu", "cuda"):
        model = Sliced(device)
        index_map = CheckpointIndexMap(model, tensors)
        for name, tensor in checkpoint.items():
            index_map.load_rows(name, tensor.view(-1, tensors[name][0][-1]).to(device), 0)
        index_map.checksums()
        maps[device] = (model, index_map)
    maps["cuda"][1].fused_tables(warm_up=True)

    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        w = torch.randperm(128, generator=generator)[:40].sort().values
        v = torch.randperm(30, generator=generator)[:7].sort().values
        s = torch.randperm(32, generator=generator)[:5].sort().values
        bf16 = torch.randint(-32768, 32768, (47,), generator=generator, dtype=torch.int32).to(torch.int16)
        fp32 = torch.randint(-(2**31), 2**31, (5,), generator=generator, dtype=torch.int64).to(torch.int32)
        for device, (_, index_map) in maps.items():
            index_map.apply_delta([("w", 40), ("v", 7)], torch.cat([w, v]).to(device), bf16.to(device))
            index_map.apply_delta([("s", 5)], s.to(device), fp32.to(device))
    (cpu_model, cpu_map), (gpu_model, gpu_map) = maps["cpu"], maps["cuda"]
    for (name, cpu_param), gpu_param in zip(cpu_model.named_parameters(), gpu_model.parameters()):
        assert torch.equal(as_bits(cpu_param.data), as_bits(gpu_param.data).cpu()), name
    running = gpu_map.checksums()[1].cpu()
    assert running.tolist() == cpu_map.checksums()[1].tolist()
    assert running.tolist() == gpu_map.checksums(full=True)[1].cpu().tolist()
