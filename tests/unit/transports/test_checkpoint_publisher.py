"""Checkpoint publisher: reading safetensors checkpoints into host buffers, splitting them between ranks, and what a
checkpoint version carries about the integer tables it leaves out."""

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.transports.weights.checkpoint_publisher import (
    HostTensors,
    MemoryFloor,
    published_tensors,
    read_checkpoint,
    read_integer_tables,
    split_tensors,
)
from prime_rl.transports.weights.checkpoint_versions import (
    checkpoint_version_uid,
    integer_tables_digest,
    version_integer_tables,
)

NO_FLOOR = MemoryFloor(gib=0, fraction=0)


def checkpoint(path, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "embed.weight": torch.randn(64, 16, generator=generator).to(torch.bfloat16),
        "layers.0.norm.weight": torch.randn(16, generator=generator),
        "layers.0.ffn.gate.tid2eid": torch.randint(0, 8, (64, 2), generator=generator),
        "layers.0.scale": torch.randn((), generator=generator).half(),
    }
    path.mkdir(exist_ok=True)
    save_file({name: tensors[name] for name in ("embed.weight", "layers.0.ffn.gate.tid2eid")}, path / "a.safetensors")
    save_file({name: tensors[name] for name in ("layers.0.norm.weight", "layers.0.scale")}, path / "b.safetensors")
    return tensors


@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_ranks_read_their_share_of_every_checkpoint_into_the_same_buffers(tmp_path, ranks):
    first = checkpoint(tmp_path / "step-1", seed=1)
    entries = published_tensors(read_checkpoint(tmp_path / "step-1"))
    assert set(entries) == {"embed.weight", "layers.0.norm.weight", "layers.0.scale"}
    shares = split_tensors(entries, ranks)
    assert sorted(name for share in shares for name in share) == sorted(entries) and all(shares)
    hosts = [HostTensors(entries, share, pin=False, floor=NO_FLOOR) for share in shares]
    pointers = [{name: t.data_ptr() for name, t in host.tensors.items()} for host in hosts]

    for step, seed in ((1, 1), (2, 2)):
        expected = first if step == 1 else checkpoint(tmp_path / "step-2", seed=seed)
        entries = read_checkpoint(tmp_path / f"step-{step}")
        for host, pointer in zip(hosts, pointers):
            assert host.check_layout(entries) == []
            host.read(entries, threads=4)
            for name, tensor in host.tensors.items():
                assert torch.equal(tensor, expected[name]) and tensor.dtype == expected[name].dtype
                assert tensor.data_ptr() == pointer[name], "ModelExpress serves the buffers in place"


def test_a_checkpoint_with_another_layout_is_refused(tmp_path):
    checkpoint(tmp_path / "step-1", seed=1)
    entries = published_tensors(read_checkpoint(tmp_path / "step-1"))
    host = HostTensors(entries, list(entries), pin=False, floor=NO_FLOOR)
    save_file({"embed.weight": torch.zeros(63, 16, dtype=torch.bfloat16)}, tmp_path / "step-1" / "a.safetensors")
    assert host.check_layout(read_checkpoint(tmp_path / "step-1")) == [
        "embed.weight: torch.bfloat16[63, 16] != torch.bfloat16[64, 16]"
    ]
    (tmp_path / "step-1" / "b.safetensors").unlink()
    assert "layers.0.norm.weight: missing" in host.check_layout(read_checkpoint(tmp_path / "step-1"))


def test_the_publisher_refuses_to_take_the_host_below_its_memory_floor(tmp_path):
    checkpoint(tmp_path, seed=1)
    entries = published_tensors(read_checkpoint(tmp_path))
    with pytest.raises(MemoryError, match="below the floor"):
        HostTensors(entries, list(entries), pin=False, floor=MemoryFloor(gib=0, fraction=1.0))


def test_floating_dtypes_modelexpress_cannot_publish_are_refused(tmp_path):
    save_file({"w": torch.zeros(4, dtype=torch.float8_e4m3fn)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="bf16 export"):
        published_tensors(read_checkpoint(tmp_path))


def test_integer_tables_digest_ignores_names_order_and_integer_width(tmp_path):
    tensors = checkpoint(tmp_path, seed=1)
    table = tensors["layers.0.ffn.gate.tid2eid"]
    digest = integer_tables_digest(read_integer_tables(read_checkpoint(tmp_path)))
    assert digest == integer_tables_digest([table.to(torch.int32)])
    assert digest != integer_tables_digest([table.roll(1, 0)])
    assert integer_tables_digest([table, table + 1]) == integer_tables_digest([table + 1, table])

    uid = checkpoint_version_uid(120, digest)
    assert uid.startswith("ckpt-120-") and version_integer_tables(uid) == digest
    assert version_integer_tables("0f3c9a12-trainer-version") is None
