import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from prime_rl.inference.vllm.worker.filesystem import _read_rows
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


def test_mapped_writes_load_patch_and_checksum_through_their_tables():
    model = Toy()
    index_map = CheckpointIndexMap(model, {"weight": ([4, 8], "bfloat16"), "norm": ([8], "bfloat16")})
    # A kernel layout with rows in the order 2, 0, 3, 1 and the two column halves swapped
    rows, cols = torch.tensor([2, 0, 3, 1]), torch.cat([torch.arange(4, 8), torch.arange(4)])
    index_map.map_write("weight", 0, row_offsets=rows * 8, col_offsets=cols)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    index_map.load_rows("weight", weight, 0)
    kernel = torch.empty_like(weight)
    kernel[rows[:, None], cols[None, :]] = weight
    assert torch.equal(model.weight.data, kernel)
    assert index_map.checksums()[1].tolist() == [bits_checksum(as_bits(weight))]

    indices = torch.tensor([1, 9, 30])
    delta = torch.tensor([5, -7, 11], dtype=torch.int16)
    index_map.apply_delta([("weight", 3)], indices, delta)
    new = as_bits(weight).clone().view(-1)
    new[indices] += delta
    kernel_bits = torch.empty_like(as_bits(weight))
    kernel_bits[rows[:, None], cols[None, :]] = new.view(4, 8)
    assert torch.equal(as_bits(model.weight.data), kernel_bits)
    assert index_map.checksums()[1].tolist() == [bits_checksum(new)]
    assert index_map.checksums(full=True)[1].tolist() == [bits_checksum(new)]


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
        # "v" through a kernel layout that reverses its rows
        index_map.map_write("v", 0, row_offsets=(5 - torch.arange(6)) * 10)
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


def test_probed_permutations_split_into_row_and_column_tables():
    from prime_rl.inference.vllm.worker.weight_transfer import _probe_permutation, _separable

    rows, cols = torch.randperm(6), torch.randperm(10)

    def kernel_layout(x):  # rows and columns permuted, into a padded [8, 12] output
        out = torch.zeros(8, 12, dtype=x.dtype)
        out[:6, :10] = x[rows][:, cols]
        return out

    where = _probe_permutation(kernel_layout, (6, 10), "cpu")
    x = torch.randint(0, 255, (6, 10), dtype=torch.uint8)
    flat = torch.zeros(96, dtype=torch.uint8)
    flat[where] = x.view(-1)
    assert torch.equal(flat, kernel_layout(x).view(-1))
    row_offsets, col_offsets = _separable(where, 6, 10)
    assert torch.equal(row_offsets[:, None] + col_offsets[None, :], where.view(6, 10))
    with pytest.raises(NotImplementedError):
        _probe_permutation(lambda x: x[:3], (6, 10), "cpu")  # drops elements


@pytest.mark.gpu
@pytest.mark.parametrize("slot", ["w13", "w2"])
def test_trtllm_mxfp8_moe_layout_tables_match_vllm(slot):
    # fused_moe first: importing flashinfer_utils on its own hits a circular import in vLLM
    import vllm.model_executor.layers.fused_moe  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_mxfp8_moe_weights,
        swap_w13_to_w31,
    )

    from prime_rl.inference.vllm.worker.weight_transfer import trtllm_mxfp8_moe_layouts
    from prime_rl.utils.sparse_weights import quantize_mx_blocks

    intermediate, hidden = 256, 512
    shapes = {"w13": (2 * intermediate, hidden), "w2": (hidden, intermediate)}
    plain = {}
    for key, (num_rows, num_cols) in shapes.items():
        quantized, scales = quantize_mx_blocks(torch.randn(num_rows, num_cols, dtype=torch.bfloat16, device="cuda"))
        plain[key], plain[f"{key}_scale"] = quantized[None], scales[None]
    w13, w13_scale = swap_w13_to_w31(plain["w13"]), swap_w13_to_w31(plain["w13_scale"])
    kernel = dict(
        zip(
            ("w13", "w2", "w13_scale", "w2_scale"),
            _shuffle_mxfp8_moe_weights(w13, plain["w2"], w13_scale, plain["w2_scale"], True),
        )
    )
    num_rows, num_cols = shapes[slot]
    matches = []
    for weight_rows, scale_rows, scale_cols in trtllm_mxfp8_moe_layouts(num_rows, num_cols, "cuda"):
        weight = torch.zeros(num_rows * num_cols, dtype=torch.uint8, device="cuda")
        weight[(weight_rows[:, None] + torch.arange(num_cols)[None, :]).cuda().view(-1)] = plain[slot][0].reshape(-1)
        scales = torch.zeros(kernel[f"{slot}_scale"].numel(), dtype=torch.uint8, device="cuda")
        scales[(scale_rows[:, None] + scale_cols[None, :]).cuda().view(-1)] = plain[f"{slot}_scale"][0].reshape(-1)
        matches.append(
            torch.equal(weight, kernel[slot].view(torch.uint8).reshape(-1))
            and torch.equal(scales, kernel[f"{slot}_scale"].view(torch.uint8).reshape(-1))
        )
    assert any(matches)


class FusedExperts(nn.Module):
    """Two experts' gate and up projections loaded into one fused ``[experts, gate; up, hidden]`` param, then, like
    FlashInfer CUTLASS's processing after loading, swapped to ``[up; gate]``."""

    def __init__(self):
        super().__init__()
        self.w13 = nn.Parameter(torch.zeros(2, 8, 4, dtype=torch.bfloat16))

    def load_weights(self, weights):
        for name, tensor in weights:
            expert, projection = int(name.split(".")[1]), name.split(".")[2]
            self.w13.data[expert, (0 if projection == "gate" else 4) : (4 if projection == "gate" else 8)].copy_(tensor)

    def swap_halves(self):
        self.w13.data.copy_(self.w13.data.view(2, 2, 4, 4).flip(1).view(2, 8, 4))


def fused_experts_boot():
    tensors = {f"experts.{e}.{p}.weight": ([4, 4], "bfloat16") for e in range(2) for p in ("gate", "up")}
    checkpoint = {name: torch.randn(4, 4, dtype=torch.bfloat16) for name in tensors}
    model = FusedExperts()
    model.load_weights(checkpoint.items())
    model.swap_halves()
    return model, tensors, checkpoint


def test_match_boot_follows_a_post_load_gate_up_swap():
    model, tensors, checkpoint = fused_experts_boot()
    index_map = CheckpointIndexMap(model, tensors)
    boot = model.w13.data.clone()
    assert index_map.match_boot((name, 0, tensor) for name, tensor in checkpoint.items()) == 4
    # Moved writes stay put on a second check, and a load through the index map keeps the engine's layout
    assert index_map.match_boot((name, 0, tensor) for name, tensor in checkpoint.items()) == 0
    for name, tensor in checkpoint.items():
        index_map.load_rows(name, tensor, 0)
    assert torch.equal(model.w13.data, boot)
    names, totals, _ = index_map.checksums()
    assert totals.tolist() == [bits_checksum(as_bits(checkpoint[name])) for name in names]


def test_match_boot_raises_on_a_block_it_cannot_find():
    model, tensors, checkpoint = fused_experts_boot()
    index_map = CheckpointIndexMap(model, tensors)
    model.w13.data[1, 0, 0] += 1
    with pytest.raises(NotImplementedError, match="experts.1.up.weight"):
        index_map.match_boot((name, 0, tensor) for name, tensor in checkpoint.items())


class TiedToy(nn.Module):
    """An engine model whose head is tied to its embedding: its loader loads both names into one param."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16))

    def load_weights(self, weights):
        for _, tensor in weights:
            self.embed.data.copy_(tensor)


def test_read_rows_skips_only_tied_names_the_checkpoint_omits(tmp_path):
    model = TiedToy()
    index_map = CheckpointIndexMap(model, {"embed": ([4, 8], "bfloat16"), "head": ([4, 8], "bfloat16")})
    assert index_map.tied == {"embed": ["head"], "head": ["embed"]}
    embed = torch.randn(4, 8, dtype=torch.bfloat16)
    # A HF checkpoint of tied embeddings may store only the embedding
    save_file({"embed": embed}, tmp_path / "model.safetensors")
    model.embed.data.copy_(embed)
    rows = list(_read_rows(index_map, tmp_path, "cpu"))
    assert [(name, begin) for name, begin, _ in rows] == [("embed", 0)]
    assert torch.equal(rows[0][2], embed)
    assert index_map.match_boot(rows) == 0

    index_map = CheckpointIndexMap(Toy(), {"weight": ([4, 8], "bfloat16"), "norm": ([8], "bfloat16")})
    save_file({"weight": embed}, tmp_path / "model.safetensors")
    with pytest.raises(KeyError, match="norm"):
        list(_read_rows(index_map, tmp_path, "cpu"))
