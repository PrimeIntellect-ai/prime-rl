import torch
from safetensors.torch import load_file, save_file

from tools.check_weight_update_bit_exact import compare, compare_tensors, failures, wire_mismatches, write_version


def entry(**changes):
    return {
        "kind": "parameter",
        "shape": [2, 2],
        "hashed_shape": [2, 2],
        "dtype": "torch.bfloat16",
        "stride": [2, 1],
        "storage_offset": 0,
        "data_ptr": 100,
        "storage_ptr": 100,
        "device": "cuda:0",
        "sha256": "w0",
        **changes,
    }


def test_comparison_detects_metadata_and_state_with_identical_bytes():
    cold = {"weight": entry(), "scale": entry(kind="buffer")}
    live = {
        "weight": entry(dtype="torch.float16", shape=[4], stride=[1]),
        "other_scale": entry(kind="buffer"),
    }
    assert compare_tensors(cold, live, None, mx=True) == {
        "compared": 1,
        "missing": ["scale"],
        "added": ["other_scale"],
        "differ": {"weight": ["dtype", "shape", "stride", "aliases"]},
    }


def test_each_install_is_checked_against_its_own_reference():
    plan = {"steps": [0, 1, 0, 2, 0], "transport": "filesystem", "forward": False}
    references = {i: {"tensors": {"weight": entry(sha256=f"w{i}")}} for i in range(3)}
    installed = {
        -1: references[0]["tensors"],
        0: references[0]["tensors"],
        1: {"weight": entry(sha256="wrong-but-different")},
        2: {"weight": entry(data_ptr=101, storage_ptr=101)},
        3: references[2]["tensors"],
        4: references[0]["tensors"],
    }
    steps = compare(plan, references, installed, {}, {}, None)
    report = {
        "error": None,
        "trainer_checkpoint_check": {"compared": 1, "differ": []},
        "reference_changes": True,
        "steps": steps,
    }
    assert failures(plan, report) == [
        "Step 1 differ: {'weight': ['sha256']}",
        "Step 2 moved_parameters: ['weight']",
    ]


def test_wire_checks_dtype_offsets_and_replica_count():
    rank0 = {"weight": ["torch.float32", 4, [4], [[[0], [2]]]]}
    rank1 = {"weight": ["torch.float32", 4, [4], [[[2], [2]]]]}
    assert wire_mismatches([rank0, rank1], [rank1, rank0])["matches"]
    wrong = {"weight": ["torch.bfloat16", 2, [4], [[[0], [2]]]]}
    for actual in ([rank0], [rank0, rank0], [rank1, wrong]):
        result = wire_mismatches([rank0, rank1], actual)
        assert not result["matches"]
        assert result["differ"] == ["weight"]


def test_checkpoint_versions_are_reproducible_and_preserve_fp32_mantissa(tmp_path):
    checkpoint = tmp_path / "w0"
    checkpoint.mkdir()
    name = "weights.safetensors"
    source = {"bias": torch.tensor([1.001, 1.002], dtype=torch.float32), "table": torch.tensor([7, 9])}
    save_file(source, checkpoint / name)
    versions = [tmp_path / f"version{i}" for i in range(3)]
    for version, seed in zip(versions, (1, 2, 1)):
        write_version(checkpoint, version, seed)
    first, second, duplicate = [load_file(path / name) for path in versions]
    assert torch.equal(first["bias"], duplicate["bias"])
    assert not torch.equal(first["bias"], second["bias"])
    assert not torch.equal(first["bias"], first["bias"].bfloat16().float())
    assert torch.equal(first["table"], source["table"])
