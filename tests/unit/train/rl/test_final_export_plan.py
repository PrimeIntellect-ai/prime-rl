"""Tests for the torch-free part of the GeneBench end-of-run export: the plan, the upload state, the restart
invalidation, the outcome markers and the per-step loop."""

import json
from pathlib import Path

import pytest

from prime_rl.trainer.rl import final_export_plan as fx
from prime_rl.utils.ckpt_retention import (
    DONE,
    RUNNING,
    Keep3Retention,
    RetentionConfig,
    record_path,
    select_keep,
)

ENV = "genebench-heldout"
TR = f"eval/{ENV}/all/agent/metrics/train_reward/mean"
HF = f"eval/{ENV}/all/agent/metrics/hurdle_frac/mean"
CFG = RetentionConfig(metric_keys={"train_reward": TR, "hurdle_frac": HF}, max_eval_error=0.05)
PREFIX = "PrimeIntellect/Nemotron-3.5-Super-genebench-rl"
RUN = "newrun0123456789abcdefgh"


def done(step, tr, hf, err=0.0, n=320):
    return {
        "env": ENV,
        "step": step,
        "status": DONE,
        "metrics": {TR: tr, HF: hf, f"eval/{ENV}/all/agent/solved_some": 0.5},
        "eval_error": err,
        "n_episodes": n,
        "policy_version": step,
    }


def running(step):
    return {"env": ENV, "step": step, "status": RUNNING}


def plan_for(records, complete, *, final_step, external=(200,), timed_out=False, tmp_path=None):
    sel = select_keep(complete, external, records, CFG)
    return fx.build_plan(
        sel,
        final_step=final_step,
        external_steps=external,
        records=records,
        eval_env=ENV,
        prefix=PREFIX,
        run_id=RUN,
        ckpt_dir=(tmp_path or Path("/volume/runs") / RUN / "checkpoints"),
        timed_out=timed_out,
        pending=sel.pending,
        now=lambda: 0.0,
    )


def test_plan_final_step_holds_every_role():
    records = {200: done(200, 0.20, 0.40), 280: done(280, 0.25, 0.45), 300: done(300, 0.30, 0.50)}
    plan = plan_for(records, [280, 300], final_step=300)
    assert plan["problems"] == [] and plan["external"] == []
    [item] = plan["exports"]
    assert item["step"] == 300 and item["source"] == fx.LIVE
    assert item["role"] == "best_train_reward+best_hurdle_frac+last"
    assert item["repo"] == f"{PREFIX}-step300"
    assert item["dcp_dir"] == f"/volume/runs/{RUN}/checkpoints/step_300/trainer"
    assert item["heldout"]["train_reward"] == 0.30 and item["heldout"]["pass_at_6"] == 0.5
    assert plan["kept_not_exported"] == {}


def test_plan_three_steps_live_first_then_descending():
    records = {
        200: done(200, 0.20, 0.40),
        240: done(240, 0.31, 0.41),  # best train_reward
        260: done(260, 0.22, 0.52),  # best hurdle_frac
        280: done(280, 0.21, 0.40),
        300: done(300, 0.20, 0.30),
    }
    plan = plan_for(records, [240, 260, 280, 300], final_step=300)
    assert [(i["step"], i["role"], i["source"]) for i in plan["exports"]] == [
        (300, "last", fx.LIVE),
        (260, "best_hurdle_frac", fx.DCP),
        (240, "best_train_reward", fx.DCP),
    ]
    assert fx.plan_steps(plan) == {240, 260, 300}


def test_plan_external_fork_step_holds_a_best_role_but_is_not_exported():
    # Round 1's step 200 (forked, external) still has the best hurdle_frac; its repo exists and is not re-uploaded.
    records = {200: done(200, 0.24, 0.43), 260: done(260, 0.26, 0.40), 300: done(300, 0.21, 0.42)}
    plan = plan_for(records, [260, 300], final_step=300)
    assert [(i["step"], i["role"]) for i in plan["exports"]] == [(300, "last"), (260, "best_train_reward")]
    [ext] = plan["external"]
    assert (ext["step"], ext["role"], ext["repo"]) == (200, "best_hurdle_frac", f"{PREFIX}-step200")
    assert "source" not in ext and "not uploaded again" in ext["note"]
    assert plan["roles"] == {"best_train_reward": 260, "best_hurdle_frac": 200, "last": 300}


def test_plan_after_eval_wait_timeout():
    # The final eval never finished: step 300 is last only, 280's eval is pending too and it stays kept.
    records = {200: done(200, 0.24, 0.43), 260: done(260, 0.26, 0.45), 280: running(280), 300: running(300)}
    plan = plan_for(records, [260, 280, 300], final_step=300, timed_out=True)
    assert [(i["step"], i["role"]) for i in plan["exports"]] == [
        (300, "last"),
        (260, "best_train_reward+best_hurdle_frac"),
    ]
    assert plan["eval_wait_timed_out"] is True
    assert plan["pending_evals"] == [280, 300]
    assert list(plan["kept_not_exported"]) == ["280"] and "eval pending" in plan["kept_not_exported"]["280"][0]
    assert plan["exports"][0]["heldout"] is None


def test_plan_failed_and_high_error_evals_cannot_be_best():
    records = {
        260: done(260, 0.40, 0.60, err=0.2),  # error rate above 5%
        280: {"env": ENV, "step": 280, "status": "failed", "reason": "no attempts returned"},
        300: done(300, 0.20, 0.30),
    }
    plan = plan_for(records, [260, 280, 300], final_step=300, external=())
    assert [(i["step"], i["role"]) for i in plan["exports"]] == [(300, "best_train_reward+best_hurdle_frac+last")]
    assert "260" in plan["ineligible"] and "280" in plan["ineligible"]


def test_plan_problem_when_last_is_not_the_final_step():
    plan = plan_for({280: done(280, 0.2, 0.3)}, [280], final_step=300, external=())
    assert plan["problems"] and "not the final step 300" in plan["problems"][0]


def test_plan_matches():
    plan = plan_for({300: done(300, 0.2, 0.3)}, [300], final_step=300)
    assert fx.plan_matches(plan, 300)
    assert not fx.plan_matches(plan, 320)
    assert not fx.plan_matches(None, 300)
    assert not fx.plan_matches({"final_step": 300}, 300)


def test_export_actions_skip_refresh_export():
    records = {240: done(240, 0.31, 0.41), 260: done(260, 0.22, 0.52), 300: done(300, 0.2, 0.3)}
    plan = plan_for(records, [240, 260, 300], final_step=300, external=())
    by_step = {i["step"]: i for i in plan["exports"]}
    state = {"uploads": {}}
    fx.record_upload(state, by_step[300], "c300", now=lambda: 0.0)
    fx.record_upload(state, dict(by_step[260], role="last"), "c260", now=lambda: 0.0)  # verified under an old role
    state["uploads"]["240"] = {"repo": "Other/repo-step240", "role": "best_train_reward", "commit": "x"}
    actions = [(i["step"], a) for i, a in fx.export_actions(plan, state)]
    assert actions == [(300, fx.SKIP), (260, fx.REFRESH), (240, fx.EXPORT)]


def test_state_roundtrip(tmp_path):
    paths = fx.ExportPaths(tmp_path)
    assert fx.load_state(paths.state) == {"uploads": {}}
    paths.root.mkdir()
    paths.state.write_text("not json")
    assert fx.load_state(paths.state) == {"uploads": {}}
    state = fx.record_upload({"uploads": {}}, {"step": 300, "repo": "r", "role": "last"}, "abc", now=lambda: 0.0)
    fx.atomic_write_json(paths.state, state)
    assert fx.load_state(paths.state)["uploads"]["300"]["commit"] == "abc"


def _seed_export_dir(paths):
    paths.root.mkdir(parents=True)
    for p in (paths.plan, paths.done, paths.failed, paths.final_checkpoints):
        p.write_text("{}")
    state = {"uploads": {str(s): {"repo": f"r{s}", "role": "last", "commit": "c"} for s in (240, 260, 300)}}
    paths.state.write_text(json.dumps(state))
    for s in (240, 260, 300):
        (paths.staging(s)).mkdir()
        (paths.staging(s) / "README.md").write_text("card")


def test_invalidate_stale_when_the_attempt_will_train(tmp_path):
    # Resumed at 260 (e.g. the final orchestrator marker was missing): 280+ are retrained, so the plan, the outcome
    # and step 300's upload no longer hold; 240 and 260 have the same weights and stay.
    paths = fx.ExportPaths(tmp_path)
    _seed_export_dir(paths)
    lines = fx.invalidate_stale(paths, 260, 300)
    assert not paths.plan.exists() and not paths.done.exists() and not paths.failed.exists()
    assert not paths.final_checkpoints.exists()
    assert sorted(fx.load_state(paths.state)["uploads"]) == ["240", "260"]
    assert paths.staging(260).exists() and not paths.staging(300).exists()
    assert any("forgot the uploads of steps [300]" in line for line in lines)


def test_invalidate_stale_fork_start_and_export_only(tmp_path):
    paths = fx.ExportPaths(tmp_path)
    _seed_export_dir(paths)
    # Resumed at max_steps: export-only, nothing is touched.
    assert fx.invalidate_stale(paths, 300, 300) == []
    assert paths.plan.exists() and len(fx.load_state(paths.state)["uploads"]) == 3
    # A fresh start (no resume step) drops everything.
    fx.invalidate_stale(paths, None, 300)
    assert fx.load_state(paths.state)["uploads"] == {} and not any(paths.root.glob("step_*"))
    # Nothing there at all is fine.
    assert fx.invalidate_stale(fx.ExportPaths(tmp_path / "empty"), 200, 300) == []


def test_outcome_markers(tmp_path):
    paths = fx.ExportPaths(tmp_path)
    assert fx.read_outcome(paths, 300) == (None, None)
    fx.write_outcome(paths, False, {"final_step": 300, "errors": {"260": "boom"}}, now=lambda: 0.0)
    status, payload = fx.read_outcome(paths, 300)
    assert status == "failed" and payload["ok"] is False and payload["errors"] == {"260": "boom"}
    assert fx.read_outcome(paths, 320) == (None, None)  # another final step's outcome does not count
    fx.write_outcome(paths, True, {"final_step": 300}, now=lambda: 0.0)
    assert fx.read_outcome(paths, 300)[0] == "done" and not paths.failed.exists()
    assert fx.read_outcome(paths)[0] == "done"


def test_clear_outcome(tmp_path):
    paths = fx.ExportPaths(tmp_path)
    assert fx.clear_outcome(paths) == []
    fx.write_outcome(paths, False, {"final_step": 300})
    assert len(fx.clear_outcome(paths)) == 1 and fx.read_outcome(paths, 300) == (None, None)


def test_export_attempts_are_capped_and_reset_by_a_new_timeline(tmp_path):
    item = {"step": 300, "role": "last", "repo": f"{PREFIX}-step300"}
    state = {"uploads": {}}
    assert fx.begin_export_attempt(state, [(item, fx.EXPORT)], 2) is None and state["export_attempts"] == 1
    assert fx.begin_export_attempt(state, [(item, fx.EXPORT)], 2) is None and state["export_attempts"] == 2
    gave_up = fx.begin_export_attempt(state, [(item, fx.EXPORT)], 2)
    assert "2 earlier export attempt(s)" in gave_up and state["export_attempts"] == 2
    # Nothing left to upload: not an attempt, never refused.
    assert fx.begin_export_attempt(state, [(item, fx.SKIP)], 2) is None and state["export_attempts"] == 2
    # An attempt that trains again (a new timeline) resets the count.
    paths = fx.ExportPaths(tmp_path)
    paths.root.mkdir(parents=True)
    fx.atomic_write_json(paths.state, state)
    lines = fx.invalidate_stale(paths, 260, 300)
    assert "export_attempts" not in fx.load_state(paths.state) and any("attempt count" in line for line in lines)


@pytest.mark.parametrize(
    "who, problem, warning",
    [
        ({"name": "me", "orgs": [{"name": "PrimeIntellect"}], "auth": {"accessToken": {"role": "write"}}}, None, None),
        ({"name": "me", "orgs": [{"name": "Other"}]}, "cannot write to 'PrimeIntellect'", None),
        ({"name": "me", "orgs": [{"name": "PrimeIntellect"}], "auth": {"accessToken": {"role": "read"}}}, "read token", None),
        ({"name": "me", "orgs": [{"name": "PrimeIntellect", "roleInOrg": "read"}]}, "only read access", None),
        (
            {"name": "me", "orgs": [{"name": "PrimeIntellect"}], "auth": {"accessToken": {"role": "fineGrained",
             "fineGrained": {"scoped": [{"entity": {"type": "org", "name": "PrimeIntellect"},
                                         "permissions": ["repo.content.read", "repo.write"]}]}}}},
            None,
            None,
        ),
        (
            {"name": "me", "orgs": [{"name": "PrimeIntellect"}], "auth": {"accessToken": {"role": "fineGrained",
             "fineGrained": {"scoped": [{"entity": {"name": "PrimeIntellect"}, "permissions": ["repo.content.read"]}]}}}},
            None,
            "no repo.write scope",
        ),
    ],
)  # fmt: skip
def test_token_write_problem(who, problem, warning):
    got_problem, got_warning = fx.token_write_problem(who, "PrimeIntellect")
    assert (got_problem is None) == (problem is None) and (problem is None or problem in got_problem)
    assert (got_warning is None) == (warning is None) and (warning is None or warning in got_warning)


def test_armed(tmp_path):
    paths = fx.ExportPaths(tmp_path)
    fx.write_armed(paths, {"final_step": 300}, now=lambda: 0.0)
    assert fx.read_json(paths.armed)["final_step"] == 300


def test_wait_for_success_and_timeout():
    clock = [0.0]
    sleeps, waits = [], []

    def sleep(s):
        sleeps.append(s)
        clock[0] += s

    calls = iter([False, False, True])
    assert fx.wait_for(
        lambda: next(calls), timeout_s=100, poll_s=30, on_wait=waits.append, sleep=sleep, now=lambda: clock[0]
    )
    assert sleeps == [30, 30] and waits == [0.0, 30.0]

    clock[0], sleeps[:] = 0.0, []
    assert not fx.wait_for(lambda: False, timeout_s=70, poll_s=30, sleep=sleep, now=lambda: clock[0])
    assert sleeps == [30, 30, 10]  # the last sleep stops at the deadline


def test_run_actions_continues_after_a_failed_step():
    items = [{"step": s, "role": "last" if s == 300 else "best_train_reward", "repo": f"r{s}"} for s in (300, 260, 240)]
    actions = [(items[0], fx.EXPORT), (items[1], fx.EXPORT), (items[2], fx.REFRESH)]
    verified, calls = [], []

    def export(item):
        calls.append(("export", item["step"]))
        if item["step"] == 300:
            raise RuntimeError("[step_300] shard upload failed on 1 rank(s)")
        return f"c{item['step']}"

    def refresh(item):
        calls.append(("refresh", item["step"]))
        return "r240"

    errors = fx.run_actions(
        actions, export=export, refresh=refresh, on_verified=lambda i, c: verified.append((i["step"], c))
    )
    assert calls == [("export", 300), ("export", 260), ("refresh", 240)]
    assert verified == [(260, "c260"), (240, "r240")]
    assert list(errors) == [300] and "shard upload failed" in errors[300]
    # Skipped steps are not run at all.
    assert fx.run_actions([(items[0], fx.SKIP)], export=export, refresh=refresh, on_verified=None) == {}


def test_final_checkpoints_doc_and_outcome_payload():
    records = {200: done(200, 0.24, 0.43), 260: done(260, 0.26, 0.40), 300: done(300, 0.21, 0.42)}
    plan = plan_for(records, [260, 300], final_step=300)
    plan["external"][0]["repo_check"] = {"exists": True, "sha": "850246e7"}
    state = fx.record_upload({"uploads": {}}, plan["exports"][0], "c300", now=lambda: 0.0)
    doc = fx.final_checkpoints_doc(plan, state, {260: "RuntimeError: boom"})
    assert [(c["step"], c["source"], c.get("commit"), c.get("error")) for c in doc["checkpoints"]] == [
        (200, "external", None, None),
        (260, fx.DCP, None, "RuntimeError: boom"),
        (300, fx.LIVE, "c300", None),
    ]
    assert doc["checkpoints"][0]["repo_check"]["sha"] == "850246e7"
    payload = fx.outcome_payload(plan, state, {260: "RuntimeError: boom"})
    assert payload["final_step"] == 300 and payload["errors"] == {"260": "RuntimeError: boom"}
    assert payload["repos"]["300"] == {"repo": f"{PREFIX}-step300", "commit": "c300", "roles": ["last"]}
    assert payload["repos"]["200"]["roles"] == ["best_hurdle_frac"]


def test_card_eval_desc_and_heldout():
    rec = done(300, 0.2, 0.3, err=0.0125, n=316)
    assert fx.card_eval_desc(rec, 16) == ("316 graded episodes, 16 per task, error rate 0.013", "pass@16")
    assert fx.card_eval_desc(None, 16) == ("not evaluated", "pass@k")
    assert fx.heldout_from_record(running(300), ENV) is None
    assert fx.heldout_from_record(rec, ENV)["hurdle_frac"] == 0.3


def make_step(ckpt_dir, step, orchestrator=True):
    d = ckpt_dir / f"step_{step}"
    (d / "trainer").mkdir(parents=True, exist_ok=True)
    (d / "trainer" / ".metadata").write_text("m")
    if orchestrator:
        (d / "orchestrator").mkdir(parents=True, exist_ok=True)
        (d / "orchestrator" / "progress.pt").write_text("p")


def write_record(ckpt_dir, rec):
    path = record_path(ckpt_dir, rec["env"], rec["step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))


@pytest.mark.parametrize("final_eval_lands", [True, False])
def test_final_settle_against_a_checkpoint_dir(tmp_path, final_eval_lands):
    """The trainer master's final settle: the final step has only its trainer marker (the orchestrator writes its
    progress.pt after the final eval), evals land while it waits, then the final prune leaves the role holders."""
    ckpt_dir = tmp_path / "checkpoints"
    for step in (240, 260, 280):
        make_step(ckpt_dir, step)
    make_step(ckpt_dir, 300, orchestrator=False)
    write_record(ckpt_dir, done(200, 0.24, 0.43))
    write_record(ckpt_dir, done(240, 0.30, 0.40))
    write_record(ckpt_dir, done(260, 0.22, 0.41))
    write_record(ckpt_dir, running(280))
    write_record(ckpt_dir, running(300))
    retention = Keep3Retention(ckpt_dir, cfg=CFG, eval_env=ENV, external_steps={200})

    landed = iter([None, (280, done(280, 0.21, 0.39)), (300, done(300, 0.20, 0.42)) if final_eval_lands else None])

    def evals_done():
        nxt = next(landed, None)
        if nxt is not None:
            write_record(ckpt_dir, nxt[1])
        result = retention.run(300, assume_complete=[300])
        return not result.selection.pending

    clock = [0.0]
    finished = fx.wait_for(
        evals_done,
        timeout_s=200,
        poll_s=30,
        sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        now=lambda: clock[0],
    )
    assert finished is final_eval_lands
    _, sel = retention.select([300], 300)
    records = {s: json.loads(record_path(ckpt_dir, ENV, s).read_text()) for s in (200, 240, 260, 280, 300)}
    plan = fx.build_plan(
        sel,
        final_step=300,
        external_steps={200},
        records=records,
        eval_env=ENV,
        prefix=PREFIX,
        run_id=RUN,
        ckpt_dir=ckpt_dir,
        timed_out=not finished,
        pending=sel.pending,
    )
    retention.protected_steps |= fx.plan_steps(plan)
    retention.run(300, assume_complete=[300])

    def present():
        return sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*"))

    if final_eval_lands:
        # 260 and 280 lost every role once 280's and 300's evals landed; 200 (external) keeps best_hurdle_frac.
        assert [(i["step"], i["role"]) for i in plan["exports"]] == [(300, "last"), (240, "best_train_reward")]
        assert [(i["step"], i["role"]) for i in plan["external"]] == [(200, "best_hurdle_frac")]
    else:
        # 300's eval never came: it is last only; 280 was judged meanwhile.
        assert plan["eval_wait_timed_out"] and plan["pending_evals"] == [300]
        assert [(i["step"], i["role"]) for i in plan["exports"]] == [(300, "last"), (240, "best_train_reward")]
    # 280 has no role but is still what a restart resumes from (300 lacks its orchestrator marker): kept.
    assert present() == [240, 280, 300]
    assert plan["kept_not_exported"] == {"280": [sel.reasons[280][0]]}
    # The orchestrator's teardown checkpoint of 300 lands; the prune after the uploads drops 280.
    (ckpt_dir / "step_300" / "orchestrator").mkdir(exist_ok=True)
    (ckpt_dir / "step_300" / "orchestrator" / "progress.pt").write_text("p")
    retention.run(300, assume_complete=[300])
    assert present() == [240, 300]
