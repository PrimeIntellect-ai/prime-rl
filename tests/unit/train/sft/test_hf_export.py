"""CPU tests of the end-of-run HF export on a synthetic text-layout checkpoint: two ranks, each with its own "node"
output directory, sharded (DTensor) weights. Fine-tuned tensors take the base dtypes, the untrained MTP tensor and
the shard layout stay the base's, the checks catch a broken mapping, and nothing raises."""

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from prime_rl.trainer.sft import hf_export
from prime_rl.trainer.world import reset_world

BASE = {
    "model-00001-of-00003.safetensors": {
        "backbone.embeddings.weight": (torch.bfloat16, (8, 4)),
        "backbone.layers.0.mixer.A_log": (torch.float32, (4,)),
    },
    "model-00002-of-00003.safetensors": {
        "backbone.layers.0.mixer.experts.0.up_proj.weight": (torch.bfloat16, (6, 4)),
        "mtp.layers.0.eh_proj.weight": (torch.bfloat16, (4, 8)),
    },
    "model-00003-of-00003.safetensors": {
        "lm_head.weight": (torch.bfloat16, (8, 4)),
    },
}


def make_base(base_dir: Path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    base_dir.mkdir(parents=True)
    tensors, weight_map = {}, {}
    for name, entries in BASE.items():
        shard = {key: torch.randn(shape).to(dtype) for key, (dtype, shape) in entries.items()}
        save_file(shard, base_dir / name, metadata={"format": "pt"})
        tensors.update(shard)
        weight_map.update({key: name for key in shard})
    (base_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (base_dir / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"]}))
    (base_dir / "tokenizer.json").write_text("{}")
    (base_dir / "unrelated-notes.txt").write_text("not part of the export")
    return tensors


class FakeLM(nn.Module):
    """Parameters named like the HF keys (identity conversion); A_log travels in fp32 like NemotronH's."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        super().__init__()
        for key, value in tensors.items():
            *path, leaf = key.split(".")
            module = self
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, nn.Parameter(value))

    @staticmethod
    def keep_in_fp32_for_weight_transfer(name: str) -> bool:
        return name.endswith("A_log")

    def convert_to_hf(self, state_dict):
        return dict(state_dict)


def trained_weights(base: dict[str, torch.Tensor], drop: str | None = None) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in base.items():
        if key.startswith("mtp.") or key == drop:
            continue
        out[key] = value.float() * 1.01 + 0.001
    return out


def _worker(rank: int, world_size: int, port: int, tmp: str, drop: str | None, queue) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    reset_world()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

        tmp = Path(tmp)
        base = {}
        for name in BASE:
            with safe_open(tmp / "base" / name, "pt") as f:
                base.update({key: f.get_tensor(key) for key in f.keys()})
        model = FakeLM(trained_weights(base, drop))
        mesh = init_device_mesh("cpu", (world_size,))
        for module in model.modules():
            for leaf, param in list(module.named_parameters(recurse=False)):
                setattr(module, leaf, nn.Parameter(distribute_tensor(param.data, mesh, [Shard(0)])))
        config = SimpleNamespace(
            model=SimpleNamespace(name=str(tmp / "base")),
            hf_export=SimpleNamespace(
                base_dir=None, keep_base_prefixes=["mtp."], max_relative_change=0.2, private=True, model_card="Test."
            ),
        )
        # Each rank's "node" has its own copy of the run directory.
        ctx = hf_export.prepare(config, tmp / f"node{rank}" / "hf-export")
        info = hf_export.ExportInfo(name="last", repo_id=None, step=7, max_steps=7, val_loss=0.5, run_id="run")
        out = hf_export.export_weights(model, info, ctx)
        queue.put((rank, str(out) if out else None, ctx.my_files))
    finally:
        dist.destroy_process_group()


def run_export(tmp_path: Path, drop: str | None = None) -> list:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(2, port, str(tmp_path), drop, queue), nprocs=2, join=True)
    return sorted(queue.get() for _ in range(2))


def test_export_two_nodes(tmp_path):
    base = make_base(tmp_path / "base")
    results = run_export(tmp_path)
    assert all(out is not None for _, out, _ in results)
    written: dict[str, dict] = {}
    for rank, out, files in results:
        out = Path(out)
        # A rank's node holds only the shards its ranks wrote (+ the master's small files).
        assert sorted(p.name for p in out.glob("*.safetensors")) == sorted(files)
        for name in files:
            with safe_open(out / name, "pt") as f:
                assert f.metadata() == {"format": "pt"}
                written[name] = {key: f.get_tensor(key) for key in f.keys()}
    assert set(written) == set(BASE)
    expected = trained_weights(base)
    for name, entries in BASE.items():
        assert set(written[name]) == set(entries)
        for key, (dtype, shape) in entries.items():
            tensor = written[name][key]
            assert tensor.dtype == dtype and tuple(tensor.shape) == shape, key
            if key.startswith("mtp."):
                assert torch.equal(tensor, base[key])  # untrained, copied as is
            else:
                wire = torch.float32 if key.endswith("A_log") else torch.bfloat16
                assert torch.equal(tensor, expected[key].to(wire).to(dtype)), key

    master = Path(results[0][1])
    report = json.loads((master / "export_report.json").read_text())
    assert report["problems"] == [] and report["tensors"] == 5 and report["fine_tuned_tensors"] == 4
    assert set(report["sha256"]) == set(BASE)
    assert sorted(p.name for p in master.iterdir() if not p.name.endswith(".safetensors")) == [
        "README.md",
        "config.json",
        "export_report.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]
    assert "weights after 7 optimizer steps of 7" in (master / "README.md").read_text()


def test_broken_mapping_fails_without_raising(tmp_path):
    make_base(tmp_path / "base")
    results = run_export(tmp_path, drop="lm_head.weight")
    assert [out for _, out, _ in results] == [None, None]
    report = json.loads((Path(tmp_path) / "node0" / "hf-export" / "last" / "export_report.json").read_text())
    assert any("lm_head.weight" in problem for problem in report["problems"])


def test_file_logic(tmp_path):
    base = make_base(tmp_path / "base")
    name = "model-00002-of-00003.safetensors"
    fine_tuned = {
        "backbone.layers.0.mixer.experts.0.up_proj.weight": base["backbone.layers.0.mixer.experts.0.up_proj.weight"]
        + 0.01
    }
    changes = hf_export.write_file(tmp_path / "base" / name, tmp_path / name, fine_tuned, ("mtp.",))
    assert not fine_tuned and list(changes) == ["backbone.layers.0.mixer.experts.0.up_proj.weight"]
    hf_export.check_layout(tmp_path / "base" / name, tmp_path / name)
    with pytest.raises(ValueError, match="key set differs"):
        save_file({"x": torch.zeros(1)}, tmp_path / "bad.safetensors")
        hf_export.check_layout(tmp_path / "base" / name, tmp_path / "bad.safetensors")
    with pytest.raises(ValueError, match="non-finite"):
        hf_export.write_file(
            tmp_path / "base" / name,
            tmp_path / name,
            {"backbone.layers.0.mixer.experts.0.up_proj.weight": torch.full((6, 4), float("nan"))},
            ("mtp.",),
        )
    # The 63 text-layout shards of Nemotron 3.5 Super over 32 ranks: every file once, at most 2 per rank.
    sizes = {f"model-{i:05d}-of-00063.safetensors": 3_980_000_000 + i for i in range(1, 64)}
    assigned = hf_export.assign_files(sizes, 32)
    assert sorted(n for names in assigned for n in names) == sorted(sizes) and max(map(len, assigned)) == 2
    summary = hf_export.summarize_changes({"a.layers.0.mixer.experts.0.w": 0.01, "lm_head.weight": 0.3})
    assert hf_export.find_problems(summary, 2, 1, 1, 0.2) == [
        f"largest relative change {('lm_head.weight', 0.3)} > 0.2"
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
