"""End-to-end test of one step's export (hf_export.prepare_context + export_one) in a one-rank gloo world: a tiny
model, a tiny base checkpoint and a fake Hugging Face Hub. The modules hf_export imports only to build or load a
real model are stubbed, so it runs on a CPU-only torch."""

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
safetensors_torch = pytest.importorskip("safetensors.torch")

from prime_rl.trainer.sft import rl_export_plan as plan  # noqa: E402

RUN = "newrun0123456789abcdef"
IN_PROJ = "backbone.layers.0.mixer.in_proj.weight"
BIAS = "backbone.layers.1.mixer.gate.e_score_correction_bias"
MTP = "mtp.layers.0.mixer.q_proj.weight"


class TinyModel(torch.nn.Module):
    """Two trained tensors under prime-rl names; convert_to_hf maps them to the base's HF names."""

    NAMES = {"w": IN_PROJ, "b": BIAS}

    def __init__(self, base: dict):
        super().__init__()
        self.w = torch.nn.Parameter(base[IN_PROJ].float() + 0.01 * torch.randn(8, 4))
        self.b = torch.nn.Parameter(base[BIAS].clone() + 0.01)

    def convert_to_hf(self, state_dict):
        return {self.NAMES[k]: v for k, v in state_dict.items()}


class FakeHub:
    """huggingface_hub.HfApi with repos held in memory ({path: sha256})."""

    repos: dict[str, dict[str, str]] = {}

    def whoami(self):
        return {"name": "tester"}

    def create_repo(self, repo_id, private=None, exist_ok=False):
        assert private is True and exist_ok is True
        FakeHub.repos.setdefault(repo_id, {".gitattributes": ""})

    def repo_info(self, repo_id, files_metadata=False):
        siblings = [SimpleNamespace(rfilename=name) for name in FakeHub.repos[repo_id]]
        return SimpleNamespace(private=True, sha=f"head{len(FakeHub.repos[repo_id])}", siblings=siblings)

    def preupload_lfs_files(self, repo_id, additions):
        assert all(op.path_in_repo.endswith(".safetensors") for op in additions)

    def create_commit(self, repo_id, operations, commit_message):
        for op in operations:
            FakeHub.repos[repo_id][op.path_in_repo] = hashlib.sha256(Path(op.path_or_fileobj).read_bytes()).hexdigest()
        return SimpleNamespace(oid="commit")

    def get_paths_info(self, repo_id, paths, expand=True):
        return [
            SimpleNamespace(path=p, lfs=SimpleNamespace(sha256=FakeHub.repos[repo_id][p]))
            for p in paths
            if p.endswith(".safetensors") and p in FakeHub.repos[repo_id]
        ]


@pytest.fixture
def hf_export(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    stubs = {
        "prime_rl.configs.sft": {"SFTConfig": object},
        "prime_rl.trainer.ckpt": {"AppState": object},
        "prime_rl.trainer.model": {"setup_model": None},
        "prime_rl.trainer.parallel_dims": {"ParallelDims": object},
        "prime_rl.trainer.utils": {"get_dcp_group": None},
        "prime_rl.utils.weights": {"resolve_fqn": lambda model, key: key},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "prime_rl.trainer.sft.hf_export", raising=False)
    module = importlib.import_module("prime_rl.trainer.sft.hf_export")
    monkeypatch.setattr(module, "EXPECTED_TENSORS", 3)
    monkeypatch.setattr(module, "EXPECTED_UNTRAINED", 1)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub)
    FakeHub.repos = {}
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield module
    dist.destroy_process_group()
    sys.modules.pop("prime_rl.trainer.sft.hf_export", None)


def make_base(base_dir: Path, hf_export) -> dict:
    base_dir.mkdir()
    tensors = {
        IN_PROJ: torch.randn(8, 4).to(torch.bfloat16),
        BIAS: torch.randn(16, dtype=torch.float32),
        MTP: torch.randn(4, 4).to(torch.bfloat16),
    }
    safetensors_torch.save_file({IN_PROJ: tensors[IN_PROJ], MTP: tensors[MTP]}, base_dir / "model-1.safetensors")
    safetensors_torch.save_file({BIAS: tensors[BIAS]}, base_dir / "model-2.safetensors")
    weight_map = {IN_PROJ: "model-1.safetensors", MTP: "model-1.safetensors", BIAS: "model-2.safetensors"}
    (base_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in hf_export.BASE_FILES:
        if not (base_dir / name).exists():
            (base_dir / name).write_text("{}")
    return tensors


def test_export_one_from_the_live_model(tmp_path, hf_export):
    base = make_base(tmp_path / "base", hf_export)
    model = TinyModel(base)
    group = dist.new_group(backend="gloo")
    ctx = hf_export.prepare_context(group, tmp_path / "base", cast_on_gpu=True)
    assert ctx.wanted == {IN_PROJ, BIAS} and ctx.num_trained == 2
    assert ctx.dtypes == {IN_PROJ: torch.bfloat16, BIAS: torch.float32}

    repo = plan.expected_repo_id(300, "Org/Model-rl2")
    spec = plan.ExportSpec(
        RUN, 300, repo, "best_train_reward+last", dcp_path=tmp_path / "ckpt" / "step_300" / "trainer"
    )
    commit, entry = hf_export.export_one(
        model,
        spec,
        ctx,
        tmp_path / "hf-export",
        group,
        load_dcp=False,
        heldout={"train_reward": 0.3, "hurdle_frac": 0.5, "pass": 0.1, "pass_at_6": 0.4},
        card_kwargs={"pass_at_label": "pass@16", "training": "GRPO, 256 rollouts per step"},
        entry_extra={"roles": ["best_train_reward", "last"]},
        preflight=True,
        delete_shards=True,
    )
    out = tmp_path / "hf-export" / "step_300"
    assert commit == f"head{len(FakeHub.repos[repo])}" and entry["commit"] == commit
    assert (
        entry["swapped_tensors"] == 2
        and entry["copied_tensors"] == 1
        and entry["roles"] == ["best_train_reward", "last"]
    )
    assert entry["source"]["dcp_dir"] == str(tmp_path / "ckpt" / "step_300" / "trainer")
    # Shards were verified on the (fake) hub, then removed locally; the small files stay.
    assert not list(out.glob("*.safetensors"))
    assert {"model-1.safetensors", "model-2.safetensors", "README.md", "export_report.json"} <= set(FakeHub.repos[repo])
    report = json.loads((out / "export_report.json").read_text())
    assert report["role"] == "best_train_reward+last" and report["source"]["run_id"] == RUN
    card = (out / "README.md").read_text()
    assert "which also has its best held-out mean train_reward" in card and "pass@16 0.400" in card

    # A second export of the same run and step into the now non-empty repo passes the guard (an earlier attempt).
    report_text = (out / "export_report.json").read_text()

    def hf_hub_download(self, repo_id, filename, revision=None, local_dir=None):
        path = Path(local_dir, filename)
        path.write_text(report_text)
        return str(path)

    FakeHub.hf_hub_download = hf_hub_download
    try:
        hf_export.export_one(
            model, spec, ctx, tmp_path / "hf-export", group, load_dcp=False, heldout=None, preflight=True
        )
        # Another run's export in the repo is refused, on every rank together.
        other = plan.ExportSpec("otherrun0123456789", 300, repo, "last")
        with pytest.raises(RuntimeError, match="Hugging Face preflight failed"):
            hf_export.export_one(
                model, other, ctx, tmp_path / "hf-export", group, load_dcp=False, heldout=None, preflight=True
            )
    finally:
        del FakeHub.hf_hub_download


def test_export_one_writes_the_bytes_a_cpu_rounding_would(tmp_path, hf_export):
    """Rounding on the device before the host copy (cast_on_gpu) gives the same shard bytes as rounding fp32 later."""
    base = make_base(tmp_path / "base", hf_export)
    model = TinyModel(base)
    group = dist.new_group(backend="gloo")
    digests = {}
    for cast in (True, False):
        ctx = hf_export.prepare_context(group, tmp_path / "base", cast_on_gpu=cast)
        spec = plan.ExportSpec(RUN, 300, plan.expected_repo_id(300, f"Org/Cast{cast}"), "last")
        _, entry = hf_export.export_one(
            model, spec, ctx, tmp_path / f"out-{cast}", group, load_dcp=False, heldout=None, preflight=True
        )
        digests[cast] = entry["sha256"]
    assert digests[True] == digests[False]


def test_export_one_refuses_a_broken_conversion(tmp_path, hf_export):
    base = make_base(tmp_path / "base", hf_export)
    model = TinyModel(base)
    model.w.data += 10.0  # a wrong key mapping looks like a huge move
    group = dist.new_group(backend="gloo")
    ctx = hf_export.prepare_context(group, tmp_path / "base", cast_on_gpu=True)
    spec = plan.ExportSpec(RUN, 300, plan.expected_repo_id(300, "Org/Model-rl2"), "last")
    with pytest.raises(RuntimeError, match="export check FAILED"):
        hf_export.export_one(model, spec, ctx, tmp_path / "out", group, load_dcp=False, heldout=None)
    assert "Org/Model-rl2-step300" not in FakeHub.repos  # nothing uploaded
