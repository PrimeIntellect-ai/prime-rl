"""Tests for the RL trainer's end-of-run export driver (trainer/rl/final_export.py) in a one-rank gloo world.

The heavy per-step export (trainer/sft/hf_export.py: DCP load, gather, shard rewrite, upload) is replaced by a
fake module, and huggingface_hub.HfApi by a fake; everything else is real: the c10d store wait, the object
broadcasts, keep-3 retention on a checkpoint directory, the plan, state and outcome files."""

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from prime_rl.trainer.rl import final_export_plan as fx  # noqa: E402
from prime_rl.utils.ckpt_retention import (  # noqa: E402
    DONE,
    RUNNING,
    WRITER_FILE,
    Keep3Retention,
    RetentionConfig,
    eval_results_dir,
    record_path,
)
from prime_rl.utils.gb_settings import GBSettingsError, load_gb_settings  # noqa: E402

ENV = "genebench-heldout"
TR = f"eval/{ENV}/all/agent/metrics/train_reward/mean"
HF = f"eval/{ENV}/all/agent/metrics/hurdle_frac/mean"
PREFIX = "PrimeIntellect/Nemotron-3.5-Super-genebench-rl"


def done(step, tr, hf):
    return {"env": ENV, "step": step, "status": DONE, "metrics": {TR: tr, HF: hf}, "eval_error": 0.0, "n_episodes": 320}


def write_record(ckpt_dir, rec):
    path = record_path(ckpt_dir, rec["env"], rec["step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))


def make_step(ckpt_dir, step, orchestrator=True):
    d = ckpt_dir / f"step_{step}"
    (d / "trainer").mkdir(parents=True, exist_ok=True)
    (d / "trainer" / ".metadata").write_text("m")
    if orchestrator:
        (d / "orchestrator").mkdir(parents=True, exist_ok=True)
        (d / "orchestrator" / "progress.pt").write_text("p")


class FakeHfApi:
    whoami_result = {"name": "tester", "orgs": [{"name": "PrimeIntellect"}]}
    whoami_error: Exception | None = None

    def whoami(self):
        if FakeHfApi.whoami_error is not None:
            raise FakeHfApi.whoami_error
        return FakeHfApi.whoami_result

    def repo_info(self, repo_id):
        return SimpleNamespace(sha="850246e79d28", private=True)

    def create_commit(self, repo_id, operations, commit_message):
        return SimpleNamespace(oid=f"refresh-{repo_id.rsplit('-', 1)[1]}")


class FakeHfExport(types.ModuleType):
    """Stands in for trainer/sft/hf_export.py: records the per-step exports."""

    BASE_REPO = "base/repo"
    BASE_REVISION = "fb0ee0bc6b3f"
    GLOO_TIMEOUT = __import__("datetime").timedelta(minutes=5)

    def __init__(self):
        super().__init__("prime_rl.trainer.sft.hf_export")
        self.exports: list[dict] = []
        self.contexts = 0
        self.fail_steps: set[int] = set()
        self.on_export = None

    def check_base(self, base_dir):
        if not Path(base_dir, "model.safetensors.index.json").is_file():
            raise FileNotFoundError(base_dir)

    def hub(self, what, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def run_guarded(self, what, group, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def prepare_context(self, group, base_dir, cast_on_gpu=False):
        assert cast_on_gpu
        self.contexts += 1
        return "ctx"

    def export_one(self, model, spec, ctx, out_root, group, **kwargs):
        assert ctx == "ctx"
        self.exports.append({"step": spec.step, "role": spec.role, "repo": spec.repo_id, **kwargs})
        if self.on_export is not None:
            self.on_export(spec.step)
        if spec.step in self.fail_steps:
            raise RuntimeError(f"[{spec.name}] shard upload failed on 1 rank(s)")
        out_dir = out_root / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.md").write_text("card")
        (out_dir / "export_report.json").write_text(json.dumps({"step": spec.step, "role": spec.role}))
        entry = {
            "step": spec.step,
            "role": spec.role,
            "repo": spec.repo_id,
            "commit": f"c{spec.step}",
            "tensor_count": 42683,
            "swapped_tensors": 41643,
            "shards": 63,
            "relative_change": {"all": {"p50": 1e-4, "max": 1e-3}},
            "heldout_eval": kwargs["heldout"],
        }
        return f"c{spec.step}", entry


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    if sys.platform == "darwin":  # gloo otherwise spends 5 s per group failing to resolve the hostname
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture
def driver(monkeypatch, world):
    """The final_export module, imported against the fake hf_export and HfApi."""
    import huggingface_hub

    fake = FakeHfExport()
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.sft.hf_export", fake)
    import prime_rl.trainer.sft as sft_pkg

    monkeypatch.setattr(sft_pkg, "hf_export", fake, raising=False)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    FakeHfApi.whoami_error = None
    monkeypatch.delitem(sys.modules, "prime_rl.trainer.rl.final_export", raising=False)
    module = importlib.import_module("prime_rl.trainer.rl.final_export")
    monkeypatch.setattr(module, "POLL_S", 0.01)
    monkeypatch.setattr(module, "MARKER_WAIT_S", 0.05)
    yield module, fake
    sys.modules.pop("prime_rl.trainer.rl.final_export", None)


def setup_run(tmp_path, *, final_eval=True, timeout_s="5"):
    """Round 2 at the end: forked from round 1's step 200 (external), local checkpoints 240-280, the final step 300
    saved by the trainer (its orchestrator marker comes after the final eval)."""
    out = tmp_path / "runs" / "newrun0123456789abcdef"
    ckpt_dir = out / "checkpoints"
    for step in (240, 260, 280):
        make_step(ckpt_dir, step)
    make_step(ckpt_dir, 300, orchestrator=False)
    for rec in (done(200, 0.24, 0.43), done(240, 0.30, 0.40), done(260, 0.22, 0.41), done(280, 0.21, 0.39)):
        write_record(ckpt_dir, rec)
    write_record(ckpt_dir, done(300, 0.20, 0.42) if final_eval else {"env": ENV, "step": 300, "status": RUNNING})
    writer = {"settings": {"final_export": True}, "batch_size": 256, "eval_group_size": {ENV: 16}}
    eval_results_dir(ckpt_dir).mkdir(parents=True, exist_ok=True)
    (eval_results_dir(ckpt_dir) / WRITER_FILE).write_text(json.dumps(writer))
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors.index.json").write_text("{}")
    fork_dir = tmp_path / "runs" / "il976qi0bkd8m0u24dw4rs8c" / "checkpoints" / "step_200"
    gb, _ = load_gb_settings(
        {
            "GB_KEEP3": "1",
            "GB_FINAL_EXPORT": "1",
            "GB_FINAL_EVAL_TIMEOUT_S": timeout_s,
            "GB_EXPORT_REPO_PREFIX": PREFIX,
        },
        environ={},
    )
    config = SimpleNamespace(
        output_dir=out,
        model=SimpleNamespace(name=str(base), offload="optimizer"),
        max_steps=300,
        env_vars={},
        optim=SimpleNamespace(lr=1e-6),
        resume=SimpleNamespace(dir=fork_dir),
    )
    return config, gb, ckpt_dir, fork_dir


def make_export(module, config, gb, ckpt_dir, fork_dir):
    retention = Keep3Retention(
        ckpt_dir,
        cfg=RetentionConfig(metric_keys=gb.metric_keys, max_eval_error=gb.max_eval_error),
        eval_env=gb.eval_env,
        external_steps={200},
    )
    return module.FinalExport(
        config, gb, ckpt_dir=ckpt_dir, retention=retention, external_steps={200}, fork_dir=fork_dir
    )


def optimizer_with_state():
    param = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.SGD([param], lr=0.1, momentum=0.9)
    param.grad = torch.ones(4)
    opt.step()
    assert opt.state
    return opt


def present_steps(ckpt_dir):
    return sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*"))


def write_final_orchestrator_marker(ckpt_dir, step=300):
    (ckpt_dir / f"step_{step}" / "orchestrator").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / f"step_{step}" / "orchestrator" / "progress.pt").write_text("p")


def test_end_of_run_export_then_restart_is_idempotent(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    # The orchestrator writes its teardown checkpoint of step 300 while the trainer uploads.
    fake.on_export = lambda step: write_final_orchestrator_marker(ckpt_dir)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    assert export.startup(200, fork_dir) is False  # the fork attempt trains
    paths = fx.ExportPaths(config.output_dir)
    assert fx.read_json(paths.armed)["final_step"] == 300

    opt = optimizer_with_state()
    assert export.run(torch.nn.Linear(2, 2), opt, 300) is True
    assert not opt.state  # released before the export
    # The live final step first, then the others from their checkpoints; step 200 (external) is not uploaded.
    assert [(e["step"], e["role"], e["load_dcp"]) for e in fake.exports] == [
        (300, "last", False),
        (240, "best_train_reward", True),
    ]
    assert all(e["preflight"] and e["delete_shards"] for e in fake.exports)
    card = fake.exports[0]["card_kwargs"]
    assert "continued from step 200 of run `il976qi0bkd8m0u24dw4rs8c`" in card["source"]
    assert card["pass_at_label"] == "pass@16" and "256 rollouts per step" in card["training"]
    assert present_steps(ckpt_dir) == [240, 300]  # pruned to the role holders (280 after the uploads)

    status, payload = fx.read_outcome(paths, 300)
    assert status == "done" and payload["ok"] and payload["errors"] == {}
    assert payload["repos"]["300"]["commit"] == "c300" and payload["repos"]["200"]["roles"] == ["best_hurdle_frac"]
    final = fx.read_json(paths.final_checkpoints)
    assert [(c["step"], c["source"]) for c in final["checkpoints"]] == [(200, "external"), (240, "dcp"), (300, "live")]
    assert final["checkpoints"][0]["repo_check"] == {"exists": True, "sha": "850246e79d28", "private": True}
    assert sorted(fx.load_state(paths.state)["uploads"]) == ["240", "300"]
    assert fx.read_json(paths.report)["all_done"] is True

    # The platform restarts the run after the export: the trainer resumes at max_steps and exports nothing again.
    fake.exports.clear()
    again = make_export(module, config, gb, ckpt_dir, None)  # resolve_resume: no fork on a restart
    assert again.fork_dir == fork_dir  # still named in the model cards
    assert again.startup(300, None) is True
    assert again.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert fake.exports == [] and fake.contexts == 1
    assert fx.read_outcome(paths, 300)[0] == "done"


def test_failed_step_is_loud_but_does_not_raise_and_a_restart_finishes_it(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)
    fake.fail_steps = {300}
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is False
    # 300 failed, 240 still went out (loaded from its checkpoint).
    assert [(e["step"], e["load_dcp"]) for e in fake.exports] == [(300, False), (240, True)]
    paths = fx.ExportPaths(config.output_dir)
    status, payload = fx.read_outcome(paths, 300)
    assert status == "failed" and "shard upload failed" in payload["errors"]["300"]
    assert sorted(fx.load_state(paths.state)["uploads"]) == ["240"]

    # An export-only restart retries only step 300; the live weights are step 300's again (resumed from it).
    fake.exports.clear()
    fake.fail_steps = set()
    again = make_export(module, config, gb, ckpt_dir, fork_dir)
    assert again.startup(300, None) is True
    assert again.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert [(e["step"], e["load_dcp"]) for e in fake.exports] == [(300, False)]
    assert fx.read_outcome(paths, 300)[0] == "done" and not paths.failed.exists()


def test_final_eval_timeout_exports_the_current_role_holders(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path, final_eval=False, timeout_s="0.05")
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert [(e["step"], e["role"]) for e in fake.exports] == [(300, "last"), (240, "best_train_reward")]
    plan = fx.read_json(fx.ExportPaths(config.output_dir).plan)
    assert plan["eval_wait_timed_out"] is True and plan["pending_evals"] == [300]


def test_reused_plan_steps_survive_the_final_prune(tmp_path, driver):
    """A restart reuses the earlier attempt's plan even if the records moved since; its steps are not pruned."""
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    paths = fx.ExportPaths(config.output_dir)
    paths.root.mkdir(parents=True)
    old_plan = {
        "final_step": 300,
        "exports": [
            {"step": 300, "roles": ["last"], "role": "last", "repo": f"{PREFIX}-step300", "source": "live",
             "dcp_dir": str(ckpt_dir / "step_300" / "trainer"), "heldout": None, "eval": None},
            {"step": 260, "roles": ["best_train_reward"], "role": "best_train_reward", "repo": f"{PREFIX}-step260",
             "source": "dcp", "dcp_dir": str(ckpt_dir / "step_260" / "trainer"), "heldout": None, "eval": None},
        ],
        "external": [],
        "problems": [],
    }  # fmt: skip
    paths.plan.write_text(json.dumps(old_plan))
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    assert export.startup(300, None) is True
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert [e["step"] for e in fake.exports] == [300, 260]
    assert 260 in present_steps(ckpt_dir)


def test_refresh_when_a_verified_step_changed_role(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)
    paths = fx.ExportPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    staged = paths.staging(240)
    staged.mkdir()
    (staged / "export_report.json").write_text(json.dumps({"step": 240, "role": "last"}))
    fx.atomic_write_json(
        paths.state, {"uploads": {"240": {"repo": f"{PREFIX}-step240", "role": "last", "commit": "old240"}}}
    )
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert [e["step"] for e in fake.exports] == [300]
    assert fx.load_state(paths.state)["uploads"]["240"] == {
        **fx.load_state(paths.state)["uploads"]["240"],
        "role": "best_train_reward",
        "commit": "refresh-step240",
    }
    assert json.loads((staged / "export_report.json").read_text())["role"] == "best_train_reward"
    assert "best held-out mean train_reward" in (staged / "README.md").read_text()


def test_startup_checks(tmp_path, driver):
    module, _ = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)

    config.max_steps = None
    with pytest.raises(GBSettingsError, match="needs max_steps"):
        make_export(module, config, gb, ckpt_dir, fork_dir).startup(200, fork_dir)
    config.max_steps = 200
    with pytest.raises(GBSettingsError, match="not after the forked step 200"):
        make_export(module, config, gb, ckpt_dir, fork_dir).startup(200, fork_dir)
    config.max_steps = 300

    FakeHfApi.whoami_result = {"name": "someone", "orgs": [{"name": "OtherOrg"}]}
    try:
        with pytest.raises(GBSettingsError, match="cannot write to 'PrimeIntellect'"):
            make_export(module, config, gb, ckpt_dir, fork_dir).startup(200, fork_dir)
    finally:
        FakeHfApi.whoami_result = {"name": "tester", "orgs": [{"name": "PrimeIntellect"}]}

    class Unavailable(OSError):
        response = SimpleNamespace(status_code=503)

    FakeHfApi.whoami_error = Unavailable("hub down")
    try:  # a Hub outage at startup only warns
        assert make_export(module, config, gb, ckpt_dir, fork_dir).startup(200, fork_dir) is False
    finally:
        FakeHfApi.whoami_error = None

    (Path(config.model.name) / "model.safetensors.index.json").unlink()
    with pytest.raises(GBSettingsError, match="startup check failed"):
        make_export(module, config, gb, ckpt_dir, fork_dir).startup(200, fork_dir)


def test_startup_clears_an_earlier_timeline(tmp_path, driver):
    module, _ = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    paths = fx.ExportPaths(config.output_dir)
    paths.root.mkdir(parents=True)
    fx.write_outcome(paths, True, {"final_step": 300})
    paths.plan.write_text(json.dumps({"final_step": 300, "exports": []}))
    # Resumed at 280 (the final orchestrator marker was missing): 281-300 are retrained.
    assert make_export(module, config, gb, ckpt_dir, fork_dir).startup(280, None) is False
    assert fx.read_outcome(paths, 300) == (None, None) and not paths.plan.exists()


def test_base_model_failure_at_the_end_fails_every_export_without_raising(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)

    def broken(group, base_dir, cast_on_gpu=False):
        raise RuntimeError("base model check failed on 32 rank(s)")

    fake.prepare_context = broken
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is False
    status, payload = fx.read_outcome(fx.ExportPaths(config.output_dir), 300)
    assert status == "failed" and sorted(payload["errors"]) == ["240", "300"]
    assert "base model check failed" in payload["errors"]["300"] and fake.exports == []


def test_without_the_final_orchestrator_marker_the_resume_point_is_kept(tmp_path, driver):
    """The orchestrator never wrote step 300's checkpoint: 280 is still what a restart resumes from, so it survives
    every prune (the export goes ahead; the marker wait only warns)."""
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert [e["step"] for e in fake.exports] == [300, 240]
    assert present_steps(ckpt_dir) == [240, 280, 300]
    final = fx.read_json(fx.ExportPaths(config.output_dir).final_checkpoints)
    assert list(final["kept_not_exported"]) == ["280"]


def test_export_only_restart_clears_an_earlier_outcome(tmp_path, driver):
    """A FAILED left by the attempt before must not release the orchestrator while this attempt exports again."""
    module, _ = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    write_final_orchestrator_marker(ckpt_dir)
    paths = fx.ExportPaths(config.output_dir)
    paths.root.mkdir(parents=True)
    fx.write_outcome(paths, False, {"final_step": 300, "errors": {"300": "boom"}})
    assert make_export(module, config, gb, ckpt_dir, fork_dir).startup(300, None) is True
    assert fx.read_outcome(paths, 300) == (None, None) and fx.read_json(paths.armed)["final_step"] == 300


def test_export_attempts_are_capped(tmp_path, driver):
    """Two attempts already died mid-export (no outcome): the next one writes FAILED instead of trying again."""
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    write_final_orchestrator_marker(ckpt_dir)
    paths = fx.ExportPaths(config.output_dir)
    paths.root.mkdir(parents=True)
    fx.atomic_write_json(paths.state, {"uploads": {}, "export_attempts": module.MAX_EXPORT_ATTEMPTS})
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    assert export.startup(300, None) is True
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is False
    assert fake.exports == []
    status, payload = fx.read_outcome(paths, 300)
    assert status == "failed" and "earlier export attempt" in payload["errors"]["attempts"]


def test_export_attempt_is_counted(tmp_path, driver):
    module, fake = driver
    config, gb, ckpt_dir, fork_dir = setup_run(tmp_path)
    export = make_export(module, config, gb, ckpt_dir, fork_dir)
    export.startup(200, fork_dir)
    assert export.run(torch.nn.Linear(2, 2), optimizer_with_state(), 300) is True
    assert fx.load_state(fx.ExportPaths(config.output_dir).state)["export_attempts"] == 1


def test_release_frees_the_offload_optimizers_pinned_buffers(driver):
    """CPUOffloadOptimizer keeps its pinned state buffers in _cpu_buffers too; clearing only the state frees nothing."""
    module, _ = driver
    inner = optimizer_with_state()
    wrapper = SimpleNamespace(base_optimizer=inner, _cpu_buffers={("p", "exp_avg"): torch.zeros(4)})
    module.release_optimizer_state(wrapper)
    assert not inner.state and wrapper._cpu_buffers == {}
