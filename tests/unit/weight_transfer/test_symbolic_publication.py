import ctypes
from types import SimpleNamespace

import torch

from prime_rl.trainer.models.conversion_ops import apply_tt_to_hf
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain
from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight
from prime_rl.weight_transfer.ownership import ShardCandidate, select_source_tensors
from prime_rl.weight_transfer.publication import SourceTensor, publish_hf_tensors, route_published_region
from prime_rl.weight_transfer.sharding import SourceShard, zip_source_destination
from prime_rl.weight_transfer.wire import AgentDescriptor, WeightManifest, decode_manifest, encode_manifest

EXPERTS = 4
INTERMEDIATE = 6
HIDDEN = 8


class NemotronConverter:
    def __init__(self) -> None:
        self.ops = conversion_chain(SimpleNamespace(num_hidden_layers=1))

    def convert_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return apply_tt_to_hf(state_dict, self.ops)


def make_source(name: str, tensor: torch.Tensor) -> tuple[SourceTensor, tuple[torch.Tensor, ...]]:
    split = (tensor.shape[0] + 1) // 2
    buffers = tuple(part.contiguous() for part in tensor.split(split, dim=0))
    shards: list[SourceShard] = []
    row = 0
    for agent, buffer in enumerate(buffers):
        shards.append(
            SourceShard(
                agent=agent,
                global_offset=(row,) + (0,) * (tensor.ndim - 1),
                shape=tuple(buffer.shape),
                address=buffer.data_ptr(),
                device_id=0,
            )
        )
        row += buffer.shape[0]
    return SourceTensor(name, tensor.dtype, tuple(tensor.shape), tuple(shards)), buffers


def test_nemotron_symbolic_conversion_and_mismatched_expert_pull() -> None:
    torch.manual_seed(0)
    prime = {
        "model.layers.0.mlp.experts.w1": torch.randn(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.w2": torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
        "model.layers.0.mlp.router.gate": torch.randn(EXPERTS, HIDDEN, dtype=torch.bfloat16),
    }
    sources: list[SourceTensor] = []
    keepalive: list[torch.Tensor] = []
    for name, tensor in prime.items():
        source, buffers = make_source(name, tensor)
        sources.append(source)
        keepalive.extend(buffers)

    published = publish_hf_tensors(NemotronConverter(), tuple(sources))
    by_name = {tensor.name: tensor for tensor in published}
    assert "backbone.layers.0.mixer.experts.1.up_proj.weight" in by_name
    assert "backbone.layers.0.mixer.experts.3.down_proj.weight" in by_name
    assert by_name["backbone.layers.0.mixer.experts.1.up_proj.weight"].segments[0].agent == 0
    assert by_name["backbone.layers.0.mixer.experts.3.up_proj.weight"].segments[0].agent == 1

    recorder = BakeRecorder()
    weights = {
        tensor.name: LazyWeight(
            tensor.name,
            torch.Size(tensor.shape),
            torch.bfloat16,
            torch.device("cpu"),
            recorder,
        )
        for tensor in published
    }
    local_experts = {1: 0, 3: 1}
    w13_meta = torch.empty(2, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="meta")
    w2_meta = torch.empty(2, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="meta")
    for global_expert, local_expert in local_experts.items():
        up = weights[f"backbone.layers.0.mixer.experts.{global_expert}.up_proj.weight"]
        down = weights[f"backbone.layers.0.mixer.experts.{global_expert}.down_proj.weight"]
        recorder.current = (object(), "w13")
        w13_meta[local_expert].copy_(up)
        recorder.current = (object(), "w2")
        w2_meta[local_expert].copy_(down)
    recorder.current = None

    destinations = {
        "w13": torch.zeros_like(w13_meta, device="cpu"),
        "w2": torch.zeros_like(w2_meta, device="cpu"),
    }
    for copy in recorder.copies:
        tensor = by_name[copy.src_name]
        offset, shape, stride = resolve_chain_region(tensor.shape, torch.bfloat16, copy.ops)
        source = route_published_region(tensor, region_elem_runs(offset, shape, stride), itemsize=2)
        destination = destinations[copy.param_name].as_strided(copy.shape, copy.stride, copy.offset)
        for _, source_address, destination_address, num_bytes in zip_source_destination(
            source, tensor_runs(destination)
        ):
            ctypes.memmove(destination_address, source_address, num_bytes)

    for global_expert, local_expert in local_experts.items():
        assert torch.equal(destinations["w13"][local_expert], prime["model.layers.0.mlp.experts.w1"][global_expert])
        assert torch.equal(destinations["w2"][local_expert], prime["model.layers.0.mlp.experts.w2"][global_expert])


def test_manifest_round_trip() -> None:
    tensor = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    source, keepalive = make_source("model.weight", tensor)
    published = publish_hf_tensors(NemotronConverter(), (source,))
    manifest = WeightManifest(
        session_id="session",
        epoch=1,
        model="nemotron",
        fingerprint="fingerprint",
        agents=(AgentDescriptor("trainer-0", b"metadata-0"), AgentDescriptor("trainer-1", b"metadata-1")),
        tensors=published,
    )
    assert decode_manifest(encode_manifest(manifest)) == manifest
    assert keepalive


def test_lazy_copy_records_materialized_destination_without_copying() -> None:
    recorder = BakeRecorder()
    layer = object()
    source = LazyWeight(
        "model.weight",
        torch.Size((2, 4)),
        torch.bfloat16,
        torch.device("cpu"),
        recorder,
    )
    # vLLM deliberately keeps some parameters (for example Mamba A_log) in
    # fp32 even when the checkpoint and trainer parameter are bf16.
    destination = torch.full((2, 4), 7, dtype=torch.float32)

    recorder.current = (layer, "weight")
    result = destination.copy_(source)
    recorder.current = None

    assert result is destination
    assert torch.equal(destination, torch.full_like(destination, 7))
    assert len(recorder.copies) == 1
    copy = recorder.copies[0]
    assert copy.src_name == "model.weight"
    assert copy.layer is layer
    assert copy.param_name == "weight"
    assert copy.offset == 0
    assert copy.shape == (2, 4)
    assert copy.stride == (4, 1)


def test_replica_ownership_is_deduplicated_and_balanced() -> None:
    candidates: list[ShardCandidate] = []
    for name in ("model.a", "model.b"):
        for rank in range(4):
            row = rank % 2
            candidates.append(
                ShardCandidate(
                    rank=rank,
                    name=name,
                    dtype=torch.bfloat16,
                    full_shape=(4, 8),
                    global_offset=(row * 2, 0),
                    shape=(2, 8),
                    address=10_000 * rank + 1_000 * row,
                    device_id=rank,
                )
            )

    sources = select_source_tensors(tuple(candidates), {rank: rank for rank in range(4)})
    assert len(sources) == 2
    for source in sources:
        assert len(source.shards) == 2
        assert {shard.global_offset[0] for shard in source.shards} == {0, 2}
    assert len({shard.agent for source in sources for shard in source.shards}) > 1
