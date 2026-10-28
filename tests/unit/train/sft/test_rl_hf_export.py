"""Tests for the GeneBench RL -> HF export: the GB_EXPORT_* interface, the report, and the shard rewrite."""

import json
from pathlib import Path

import pytest

from prime_rl.trainer.sft import rl_export_plan as plan

RUN = "abcd1234efgh5678ijkl9012"


def env(steps="120,200", repos=None, roles="best,last", run_id=RUN):
    if repos is None:
        repos = ",".join(plan.expected_repo_id(int(s)) for s in steps.split(","))
    out = {"GB_EXPORT_RUN_ID": run_id, "GB_EXPORT_STEPS": steps, "GB_EXPORT_REPOS": repos, "GB_EXPORT_ROLES": roles}
    return {k: v for k, v in out.items() if v is not None}


def test_parse_two_steps():
    specs = plan.parse_export_env(env())
    assert [(s.step, s.role, s.repo_id) for s in specs] == [
        (120, "best", "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step120"),
        (200, "last", "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step200"),
    ]
    assert specs[0].dcp_dir() == Path(f"/volume/runs/{RUN}/checkpoints/step_120/trainer")
    assert specs[0].name == "step_120"


def test_parse_best_is_last_and_whitespace():
    specs = plan.parse_export_env(env(steps=" 200 ", roles=" best+last"))
    assert [(s.step, s.role) for s in specs] == [(200, "best+last")]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"run_id": None}, "GB_EXPORT_RUN_ID is not set"),
        ({"run_id": "Bad-Id"}, "is not a run id"),
        ({"roles": None}, "GB_EXPORT_ROLES is not set"),
        ({"roles": "  "}, "GB_EXPORT_ROLES is not set"),
        ({"roles": "best"}, "same number of items"),
        ({"steps": "120,,200", "roles": "best,,last", "repos": "a,b,c"}, "empty item"),
        ({"steps": "0", "roles": "last"}, "not a positive integer"),
        ({"steps": "12a", "roles": "last", "repos": "PrimeIntellect/x"}, "not a positive integer"),
        ({"steps": "120,120", "roles": "best,last"}, "duplicate steps"),
        ({"roles": "best,final"}, "is not one of"),
        ({"roles": "best,best"}, "given to several steps"),
        ({"roles": "best+last,last"}, "given to several steps"),
        ({"roles": "last,best"}, "after the last step"),
        ({"steps": "120", "roles": "best"}, "last checkpoint must be exported"),
        ({"roles": "best,best+last"}, "given to several steps"),
        (
            {
                "repos": "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step200,"
                "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step120"
            },
            "step 120 must go to",
        ),
        (
            {"steps": "200", "roles": "last", "repos": "SomeoneElse/Nemotron-3.5-Super-genebench-rl-step200"},
            "must go to",
        ),
    ],
)
def test_parse_rejects(overrides, message):
    kwargs = {"steps": "120,200", "repos": None, "roles": "best,last", "run_id": RUN}
    kwargs.update(overrides)
    with pytest.raises(plan.ExportSpecError, match=message):
        plan.parse_export_env(env(**kwargs))


def test_resolve_export_env_sources():
    full = env()
    values, sources = plan.resolve_export_env({}, full)
    assert values == full and set(sources.values()) == {"config env_vars"}
    values, sources = plan.resolve_export_env(full, None)
    assert values == full and set(sources.values()) == {"environment"}
    # Split across both, and equal values in both, are fine; the environment is reported as the source.
    half = {k: full[k] for k in ("GB_EXPORT_RUN_ID", "GB_EXPORT_STEPS")}
    rest = {k: v for k, v in full.items() if k not in half}
    values, sources = plan.resolve_export_env(dict(half, PATH="/bin"), dict(rest, GB_EXPORT_STEPS=" 120,200 "))
    assert values == full and sources["GB_EXPORT_ROLES"] == "config env_vars"
    assert plan.parse_export_env(values)[1].role == "last"
    with pytest.raises(plan.ExportSpecError, match="differs between"):
        plan.resolve_export_env(full, dict(full, GB_EXPORT_STEPS="80,200"))
    # Nothing anywhere: the parser names the missing variable.
    with pytest.raises(plan.ExportSpecError, match="GB_EXPORT_RUN_ID is not set"):
        plan.parse_export_env(plan.resolve_export_env({}, {})[0])


def make_ckpts(root: Path, complete: list[int], partial: list[int] = ()) -> None:
    for step in complete:
        trainer = root / RUN / "checkpoints" / f"step_{step}" / "trainer"
        trainer.mkdir(parents=True)
        (trainer / ".metadata").write_bytes(b"x")
    for step in partial:
        (root / RUN / "checkpoints" / f"step_{step}" / "trainer").mkdir(parents=True)


def test_storage_problems(tmp_path):
    (tmp_path / "__0_0.distcp").write_bytes(b"x" * 10)
    (tmp_path / "__1_0.distcp").write_bytes(b"x" * 4)
    assert plan.storage_problems(tmp_path, {"__0_0.distcp": 10}) == []
    problems = plan.storage_problems(tmp_path, {"__0_0.distcp": 10, "__1_0.distcp": 8, "__2_0.distcp": 1})
    assert len(problems) == 2 and "has 4 bytes, .metadata needs 8" in problems[0] and "missing" in problems[1]
    assert "no data files" in plan.storage_problems(tmp_path, {})[0]


def test_check_checkpoints_uses_completeness_predicate(tmp_path):
    # Step 200's .metadata is there, but its data files are still being written: 160 is the newest complete one.
    make_ckpts(tmp_path, [40, 80, 120, 160, 200])
    still_syncing = lambda dcp_dir: dcp_dir.parent.name != "step_200"  # noqa: E731
    assert plan.complete_checkpoint_steps(tmp_path / RUN / "checkpoints", still_syncing) == [40, 80, 120, 160]
    with pytest.raises(plan.ExportSpecError, match=r"no complete trainer checkpoint.*step_200"):
        plan.check_checkpoints(plan.parse_export_env(env()), tmp_path, still_syncing)
    specs = plan.parse_export_env(env(steps="120,160"))
    assert plan.check_checkpoints(specs, tmp_path, still_syncing)[-1] == 160


def test_check_checkpoints_best_on_newest_needs_best_plus_last(tmp_path):
    make_ckpts(tmp_path, [40, 80, 120])
    specs = [plan.ExportSpec(RUN, 120, plan.expected_repo_id(120), "best")]  # bypasses the parser's last check
    with pytest.raises(plan.ExportSpecError, match="use the role 'best\\+last'"):
        plan.check_checkpoints(specs, tmp_path)
    assert plan.check_checkpoints(plan.parse_export_env(env(steps="120", roles="best+last")), tmp_path) == [40, 80, 120]


def test_check_checkpoints(tmp_path):
    make_ckpts(tmp_path, [40, 80, 120, 160, 200])
    specs = plan.parse_export_env(env())
    assert plan.check_checkpoints(specs, tmp_path) == [40, 80, 120, 160, 200]


def test_check_checkpoints_missing_and_partial(tmp_path):
    make_ckpts(tmp_path, [40, 80, 160], partial=[120])
    with pytest.raises(plan.ExportSpecError, match="no complete trainer checkpoint"):
        plan.check_checkpoints(plan.parse_export_env(env(steps="120,160")), tmp_path)


def test_check_checkpoints_last_must_be_newest(tmp_path):
    make_ckpts(tmp_path, [40, 80, 120, 160, 200])
    with pytest.raises(plan.ExportSpecError, match="newest complete checkpoint is step 200"):
        plan.check_checkpoints(plan.parse_export_env(env(steps="120,160")), tmp_path)


def test_heldout_evals_and_best(tmp_path):
    pre = plan.HELDOUT_PREFIX
    rows = [
        {"step": None, "dispatcher/mode": 1.0},
        {
            "step": 40.0,
            pre + "metrics/train_reward/mean": 0.30,
            pre + "solved_some": 0.2,
            pre + "rewards/pass/mean": 0.1,
        },
        {"step": 80.0, pre + "metrics/train_reward/mean": 0.35, pre + "solved_some": 0.2},
        {"step": 80.0, pre + "metrics/train_reward/mean": 0.40, pre + "solved_some": 0.3},  # re-run after a resume
        {"step": 120.0, pre + "metrics/train_reward/mean": 0.40, pre + "solved_some": 0.25},
        {"step": 160.0, pre + "metrics/train_reward/mean": 0.40, pre + "solved_some": 0.3},
        {"step": 160.0, "train/loss": 1.0},  # a train row at the same step does not replace the eval
        {"step": 200.0, pre + "metrics/train_reward/mean": float("nan")},
        # One eval split over two rows: the later partial row does not erase train_reward.
        {"step": 240.0, pre + "metrics/train_reward/mean": 0.1, pre + "solved_some": 0.1},
        {"step": 240.0, pre + "rewards/pass/mean": 0.05, pre + "metrics/train_reward/mean": None},
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    evals = plan.heldout_evals(path)
    assert sorted(evals) == [40, 80, 120, 160, 200, 240]
    assert evals[240] == {"train_reward": 0.1, "hurdle_frac": None, "pass": 0.05, "pass_at_6": 0.1}
    assert evals[80]["train_reward"] == 0.40 and evals[40]["pass"] == 0.1 and evals[200]["train_reward"] is None
    # 80, 120 and 160 tie on train_reward; 80 and 160 tie on pass@6; the later step wins.
    assert plan.best_step(evals, [40, 80, 120, 160, 200]) == 160
    assert plan.best_step(evals, [40, 120]) == 120
    assert plan.best_step({}, [40]) is None
    assert plan.heldout_evals(tmp_path / "missing.jsonl") == {}


def test_check_changes():
    exported = {f"backbone.layers.{i}.mixer.in_proj.weight": 0.001 for i in range(10)}
    trained = dict(exported)
    assert plan.check_changes(exported, trained, 10, 0.2) == []
    assert "swapped tensors" in plan.check_changes(exported, trained, 11, 0.2)[0]
    big = dict(exported, **{"backbone.layers.0.mixer.in_proj.weight": 0.5})
    assert "largest relative change" in plan.check_changes(big, trained, 10, 0.2)[0]
    zeros = {k: 0.0 for k in trained}
    assert "equal the base in fp32" in plan.check_changes(zeros, zeros, 10, 0.2)[0]
    # bf16 rounding can hide tiny RL updates in the exported tensors; only fp32 equality counts as "not loaded".
    assert plan.check_changes(zeros, trained, 10, 0.2) == []


def test_report_entry_done_line_and_card(tmp_path):
    spec = plan.parse_export_env(env(steps="200", roles="best+last"))[0]
    heldout = {"train_reward": 0.41, "hurdle_frac": 0.5, "pass": 0.1, "pass_at_6": 0.3}
    entry = plan.step_entry(
        spec,
        commit="abc123",
        tensor_count=42683,
        swapped=41643,
        copied=1040,
        sha256={"model-00002-of-00063.safetensors": "b" * 64, "model-00001-of-00063.safetensors": "a" * 64},
        exported_changes={"backbone.layers.0.mixer.experts.0.up_proj.weight": 0.002, "lm_head.weight": 0.0},
        trained_changes={"backbone.layers.0.mixer.experts.0.up_proj.weight": 0.0021, "lm_head.weight": 0.0001},
        heldout=heldout,
    )
    assert entry["repo"] == "PrimeIntellect/Nemotron-3.5-Super-genebench-rl-step200"
    assert (entry["role"], entry["commit"], entry["tensor_count"], entry["swapped_tensors"]) == (
        "best+last",
        "abc123",
        42683,
        41643,
    )
    assert list(entry["sha256"]) == ["model-00001-of-00063.safetensors", "model-00002-of-00063.safetensors"]
    assert entry["relative_change"]["all"]["zero"] == 1 and entry["relative_change_fp32"]["all"]["zero"] == 0
    assert set(entry["relative_change"]["groups"]) == {"routed_experts", "lm_head.weight"}

    line = plan.done_line(entry)
    marker, payload = line.split(" ", 1)
    assert marker == plan.DONE_MARKER and "\n" not in line
    summary = json.loads(payload)
    assert summary["step"] == 200 and summary["commit"] == "abc123" and summary["heldout_eval"] == heldout
    assert "sha256" not in summary

    report = tmp_path / "export_report.json"
    plan.write_json(report, {"exports": [entry]})
    assert json.loads(report.read_text())["exports"][0]["sha256"] == entry["sha256"]
    assert not report.with_name("export_report.json.tmp").exists()

    card = plan.model_card(spec, heldout, "base/repo", "fb0ee0bc6b3f")
    assert f"run `{RUN}`, step 200" in card and "which also has its best held-out eval" in card
    assert "mean train_reward 0.410" in card and "pass@6 0.300" in card
    assert "not found" in plan.model_card(spec, None, "base/repo", "fb0ee0bc6b3f")
    assert "one BF16 rounding step" in card


class FakeHTTPError(OSError):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


class ConnectError(Exception):  # stands in for httpx.ConnectError (not an OSError)
    pass


@pytest.mark.parametrize(
    "exc, transient",
    [
        (FakeHTTPError(503), True),
        (FakeHTTPError(429), True),
        (FakeHTTPError(401), False),
        (FakeHTTPError(404), False),
        (ConnectionResetError("reset"), True),
        (TimeoutError(), True),
        (ConnectError("dns"), True),
        (RuntimeError("xet upload failed"), True),
        (FileNotFoundError("x"), False),
        (ValueError("bad"), False),
    ],
)
def test_is_transient(exc, transient):
    assert plan.is_transient(exc) is transient


def test_retry_backoff_and_give_up():
    calls, sleeps, logs = [], [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise FakeHTTPError(502)
        return "ok"

    assert plan.retry(flaky, "commit", attempts=5, base_delay=1, log=logs.append, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and sleeps == [1, 2] and "retry 1/4" in logs[0]

    calls.clear()

    def always():
        calls.append(1)
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        plan.retry(always, "commit", attempts=3, base_delay=1, sleep=lambda _: None)
    assert len(calls) == 3

    calls.clear()

    def forbidden():
        calls.append(1)
        raise FakeHTTPError(403)

    with pytest.raises(FakeHTTPError):
        plan.retry(forbidden, "commit", attempts=5, base_delay=1, sleep=lambda _: None)
    assert len(calls) == 1


def test_write_file_swaps_casts_and_copies(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from safetensors.torch import save_file

    from prime_rl.trainer.sft import rl_export_files as files

    base = {
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(8, 4).to(torch.bfloat16),
        "backbone.layers.0.mixer.A_log": torch.randn(4).to(torch.bfloat16),
        "backbone.layers.1.mixer.gate.e_score_correction_bias": torch.randn(16, dtype=torch.float32),
        "mtp.layers.0.mixer.q_proj.weight": torch.randn(4, 4).to(torch.bfloat16),
    }
    (tmp_path / "base").mkdir()
    (tmp_path / "out").mkdir()
    save_file(base, tmp_path / "base" / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    trained = {k: v.float() + 1e-3 for k, v in base.items() if files.is_trained(k)}
    trained["backbone.layers.0.mixer.A_log"] = base["backbone.layers.0.mixer.A_log"].float()  # unchanged
    expected = {k: v.clone() for k, v in trained.items()}

    exported, fp32, copied, digests = files.write_my_files(
        tmp_path / "base", ["model-00001-of-00001.safetensors"], tmp_path / "out", trained
    )
    assert trained == {}  # consumed
    assert copied == 1 and set(exported) == set(fp32) == set(expected)
    assert fp32["backbone.layers.0.mixer.A_log"] == 0.0 and fp32["backbone.layers.0.mixer.in_proj.weight"] > 0
    out_path = tmp_path / "out" / "model-00001-of-00001.safetensors"
    assert digests == {"model-00001-of-00001.safetensors": files.sha256_file(out_path)}
    with safe_open(out_path, framework="pt") as f:
        assert f.metadata() == {"format": "pt"}
        for key, value in base.items():
            got = f.get_tensor(key)
            assert got.dtype == value.dtype and got.shape == value.shape
            if files.is_trained(key):
                assert torch.equal(got, expected[key].to(value.dtype))
            else:
                assert torch.equal(got, value)


def test_write_file_rejects_bad_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from prime_rl.trainer.sft import rl_export_files as files

    save_file({"backbone.x.weight": torch.zeros(2, 2)}, tmp_path / "a.safetensors")
    with pytest.raises(ValueError, match="shape"):
        files.write_file(tmp_path / "a.safetensors", tmp_path / "b.safetensors", {"backbone.x.weight": torch.zeros(3)})
    with pytest.raises(ValueError, match="non-finite"):
        files.write_file(
            tmp_path / "a.safetensors",
            tmp_path / "b.safetensors",
            {"backbone.x.weight": torch.full((2, 2), float("inf"))},
        )
    with pytest.raises(KeyError):
        files.write_file(tmp_path / "a.safetensors", tmp_path / "b.safetensors", {})
    with pytest.raises(ValueError, match="not in this rank's base files"):
        files.write_my_files(
            tmp_path, ["a.safetensors"], tmp_path, {"backbone.x.weight": torch.ones(2, 2), "extra": torch.ones(1)}
        )


def test_assign_files_balanced_and_complete():
    pytest.importorskip("torch")
    from prime_rl.trainer.sft.rl_export_files import assign_files

    sizes = {f"model-{i:05d}.safetensors": 4_000 + (i % 3) for i in range(63)}
    assignment = assign_files(sizes, 32)
    assert sorted(f for a in assignment for f in a) == sorted(sizes)
    assert sorted(len(a) for a in assignment) == [1] + [2] * 31
    assert assign_files(sizes, 32) == assignment  # deterministic across ranks
