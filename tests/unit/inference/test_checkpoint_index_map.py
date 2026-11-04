import pytest
import torch
from torch import nn

from prime_rl.inference.vllm.worker.weight_transfer import CheckpointIndexMap, expert_slots
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


NUM_EXPERTS, LOCAL_SLOTS, HIDDEN, INTERMEDIATE = 4, 3, 8, 4
# EP rank 1 of 2 with 2 redundant experts: physical slots 3, 4, 5 of EPLB's initial placement [0, 1, 2, 3, 0, 1]
INITIAL = [0, 1, 2, 3, 0, 1]
LOCAL_TO_GLOBAL = [3, 4, 5]
KINDS = {"gate_proj": [INTERMEDIATE, HIDDEN], "up_proj": [INTERMEDIATE, HIDDEN], "down_proj": [HIDDEN, INTERMEDIATE]}
# Layer 0 holds expert 2 twice on this rank and drops expert 3; layer 1 moves its experts between slots
MOVED = [[2, 3, 1, 0, 2, 2], [3, 2, 1, 1, 0, 3]]


class ToyMoE(nn.Module):
    """Two MoE layers whose loader writes each checkpoint expert into every local slot that holds it, by default of
    EPLB's initial placement like vLLM's loader (gate and up packed into ``w13``)."""

    def __init__(self):
        super().__init__()
        self.w13 = nn.ParameterList(
            nn.Parameter(torch.zeros(LOCAL_SLOTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)) for _ in range(2)
        )
        self.w2 = nn.ParameterList(
            nn.Parameter(torch.zeros(LOCAL_SLOTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)) for _ in range(2)
        )
        self.norm = nn.Parameter(torch.zeros(HIDDEN, dtype=torch.bfloat16))

    def slot(self, layer: int, kind: str, slot: int) -> torch.Tensor:
        if kind == "down_proj":
            return self.w2[layer].data[slot]
        rows = slice(0, INTERMEDIATE) if kind == "gate_proj" else slice(INTERMEDIATE, 2 * INTERMEDIATE)
        return self.w13[layer].data[slot, rows]

    def load_weights(self, weights, placement=(INITIAL, INITIAL)):
        for name, tensor in weights:
            if name == "norm":
                self.norm.data.copy_(tensor)
                continue
            _, layer, _, _, expert, kind, _ = name.split(".")
            for slot, physical in enumerate(LOCAL_TO_GLOBAL):
                if placement[int(layer)][physical] == int(expert):
                    self.slot(int(layer), kind, slot).copy_(tensor)

    def expert_weights(self) -> list[list[torch.Tensor]]:
        """Like vLLM's ``MixtureOfExperts.expert_weights``: each layer's expert params as ``[local slots, -1]``."""
        return [[self.w13[layer].view(LOCAL_SLOTS, -1), self.w2[layer].view(LOCAL_SLOTS, -1)] for layer in range(2)]


def moe_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    checkpoint = {"norm": torch.randn(HIDDEN, generator=generator).bfloat16()}
    for layer in range(2):
        for expert in range(NUM_EXPERTS):
            for kind, shape in KINDS.items():
                checkpoint[f"layers.{layer}.mlp.experts.{expert}.{kind}.weight"] = torch.randn(
                    shape, generator=generator
                ).bfloat16()
    return checkpoint


def eplb_index_map(model: ToyMoE) -> CheckpointIndexMap:
    tensors = {name: (list(tensor.shape), "bfloat16") for name, tensor in moe_checkpoint(0).items()}
    index_map = CheckpointIndexMap(model, tensors)
    slots = expert_slots(index_map.params, model.expert_weights(), [LOCAL_TO_GLOBAL] * 2)
    index_map.track_experts(slots, torch.tensor([INITIAL, INITIAL]), NUM_EXPERTS)
    return index_map


def load(index_map: CheckpointIndexMap, checkpoint: dict[str, torch.Tensor]) -> None:
    for name, (begin, end) in index_map.row_ranges().items():
        index_map.load_rows(name, checkpoint[name].view(-1, index_map.num_cols[name])[begin:end], begin)


def held_checksums(index_map: CheckpointIndexMap, checkpoint: dict[str, torch.Tensor]) -> dict[str, int]:
    """Each held tensor's checksum once per local copy, wrapping in int64 like the engine's sums."""
    expected = {}
    for name, writes in index_map.records.items():
        copies = sum(rows * cols for _, rows, _, cols, *_ in writes) // index_map.numel[name]
        checksum = torch.tensor(bits_checksum(as_bits(checkpoint[name])), dtype=torch.int64)
        expected[name] = int(checksum * copies)
    return expected


def test_retargeted_experts_load_patch_and_checksum_like_a_fresh_load():
    model = ToyMoE()
    index_map = eplb_index_map(model)
    # Slots 3, 4, 5 start out with experts 3, 0 (a redundant replica) and 1
    held = {name.split(".")[4] for name in index_map.records if name.startswith("layers.0.")}
    assert held == {"3", "0", "1"}
    checkpoint = moe_checkpoint(1)
    load(index_map, checkpoint)
    names, totals, _ = index_map.checksums()
    assert dict(zip(names, totals.tolist())) == held_checksums(index_map, checkpoint)

    # EPLB moves whole slots in place, here as if from wherever each expert was
    for layer, placement in enumerate(MOVED):
        for slot, physical in enumerate(LOCAL_TO_GLOBAL):
            for kind in KINDS:
                model.slot(layer, kind, slot).copy_(
                    checkpoint[f"layers.{layer}.mlp.experts.{placement[physical]}.{kind}.weight"]
                )
    index_map.retarget(torch.tensor(MOVED))
    assert "layers.0.mlp.experts.3.down_proj.weight" not in index_map.names
    assert len(index_map.records["layers.0.mlp.experts.2.down_proj.weight"]) == 2
    names, totals, _ = index_map.checksums()
    assert dict(zip(names, totals.tolist())) == held_checksums(index_map, checkpoint)

    # A delta reaches both replicas of layer 0's expert 2 and skips the expert this worker no longer holds
    generator = torch.Generator().manual_seed(2)
    entries, indices, deltas, new = [], [], [], dict(checkpoint)
    for name in ("layers.0.mlp.experts.2.gate_proj.weight", "layers.0.mlp.experts.3.up_proj.weight", "norm"):
        changed = torch.randperm(checkpoint[name].numel(), generator=generator)[:5].sort().values
        delta = torch.randint(-300, 300, (5,), generator=generator, dtype=torch.int16)
        bits = as_bits(checkpoint[name]).clone().view(-1)
        bits[changed] += delta
        new[name] = bits.view(torch.bfloat16).view(checkpoint[name].shape)
        entries.append((name, 5))
        indices.append(changed)
        deltas.append(delta)
    index_map.apply_delta(entries, torch.cat(indices), torch.cat(deltas))

    fresh = ToyMoE()
    fresh.load_weights(new.items(), placement=MOVED)
    for param, expected in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(as_bits(param.data), as_bits(expected.data))
    names, running, _ = index_map.checksums()
    assert dict(zip(names, running.tolist())) == held_checksums(index_map, new)
    assert running.tolist() == index_map.checksums(full=True)[1].tolist()

    # An anchor load writes every local replica of the current placement
    reloaded = ToyMoE()
    reloaded_map = eplb_index_map(reloaded)
    reloaded_map.retarget(torch.tensor(MOVED))
    load(reloaded_map, new)
    for param, expected in zip(reloaded.parameters(), fresh.parameters()):
        assert torch.equal(as_bits(param.data), as_bits(expected.data))


def test_retarget_back_restores_the_records():
    index_map = eplb_index_map(ToyMoE())
    records = {name: sorted(writes) for name, writes in index_map.records.items()}
    index_map.retarget(torch.tensor(MOVED))
    index_map.retarget(torch.tensor([INITIAL, INITIAL]))
    assert {name: sorted(writes) for name, writes in index_map.records.items()} == records
