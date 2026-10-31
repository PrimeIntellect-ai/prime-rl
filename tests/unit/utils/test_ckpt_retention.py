import json
import shutil

import pytest

from prime_rl.utils.ckpt_retention import (
    CKPT_COMPLETE_MARKERS,
    DONE,
    FAILED,
    LOST,
    RETENTION_FILE,
    RUNNING,
    SKIPPED,
    WRITER_FILE,
    EvalRecorder,
    Keep3Retention,
    RetentionConfig,
    eval_error,
    eval_results_dir,
    fork_steps,
    plan_deletions,
    read_records,
    record_path,
    scan_ckpt_dir,
    select_keep,
)

ENV = "genebench-heldout"
TR = f"eval/{ENV}/all/agent/metrics/train_reward/mean"
HF = f"eval/{ENV}/all/agent/metrics/hurdle_frac/mean"
ERR = f"eval/{ENV}/all/has_error/mean"
CFG = RetentionConfig(metric_keys={"train_reward": TR, "hurdle_frac": HF}, max_eval_error=0.05)


def done(step, tr, hf, err=0.0):
    return {"env": ENV, "step": step, "status": DONE, "metrics": {TR: tr, HF: hf}, "eval_error": err}


def status(step, s):
    return {"env": ENV, "step": step, "status": s}


def make_step(ckpt_dir, step, trainer=True, orchestrator=True):
    d = ckpt_dir / f"step_{step}"
    (d / "trainer").mkdir(parents=True, exist_ok=True)
    (d / "orchestrator").mkdir(parents=True, exist_ok=True)
    (d / "trainer" / "__0_0.distcp").write_text("x")
    if trainer:
        (d / "trainer" / ".metadata").write_text("m")
    if orchestrator:
        (d / "orchestrator" / "progress.pt").write_text("p")


def write(ckpt_dir, rec):
    path = record_path(ckpt_dir, rec["env"], rec["step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))


def test_markers_match_pathing():
    from prime_rl.utils.pathing import CKPT_COMPLETE_MARKERS as PATHING_MARKERS

    assert CKPT_COMPLETE_MARKERS == PATHING_MARKERS


# ── select_keep ────────────────────────────────────────────────────────────


def test_no_records_keeps_every_complete_step_as_pending():
    sel = select_keep([220, 240], [], {}, CFG)
    assert sel.keep == {220, 240}
    assert sel.last == 240
    assert sel.pending == [220, 240]
    assert sel.roles == {"best_train_reward": None, "best_hurdle_frac": None, "last": 240}


def test_no_checkpoints_at_all():
    sel = select_keep([], [], {}, CFG)
    assert sel.keep == set() and sel.last is None and sel.roles["last"] is None


def test_first_eval_is_best_for_both_metrics():
    sel = select_keep([220], [], {220: done(220, 0.2, 0.4)}, CFG)
    assert sel.keep == {220}
    assert sel.roles == {"best_train_reward": 220, "best_hurdle_frac": 220, "last": 220}
    assert sel.step_roles() == {220: ["best_train_reward", "best_hurdle_frac", "last"]}


def test_three_distinct_roles():
    records = {220: done(220, 0.30, 0.40), 240: done(240, 0.20, 0.50), 260: done(260, 0.10, 0.10)}
    sel = select_keep([220, 240, 260], [], records, CFG)
    assert sel.keep == {220, 240, 260}
    assert sel.roles == {"best_train_reward": 220, "best_hurdle_frac": 240, "last": 260}


def test_non_best_non_last_step_is_dropped():
    records = {220: done(220, 0.30, 0.50), 240: done(240, 0.20, 0.40), 260: done(260, 0.10, 0.10)}
    sel = select_keep([220, 240, 260], [], records, CFG)
    assert sel.keep == {220, 260}


def test_ties_go_to_the_later_step():
    records = {220: done(220, 0.25, 0.40), 240: done(240, 0.25, 0.40), 260: done(260, 0.10, 0.10)}
    sel = select_keep([220, 240, 260], [], records, CFG)
    assert sel.roles["best_train_reward"] == 240
    assert sel.roles["best_hurdle_frac"] == 240
    assert sel.keep == {240, 260}


def test_failed_lost_skipped_and_high_error_evals_cannot_be_best():
    records = {
        220: done(220, 0.90, 0.90, err=0.06),
        240: status(240, FAILED),
        260: status(260, LOST),
        280: status(280, SKIPPED),
        300: done(300, 0.10, 0.10, err=0.05),
    }
    sel = select_keep([220, 240, 260, 280, 300], [], records, CFG)
    assert sel.roles == {"best_train_reward": 300, "best_hurdle_frac": 300, "last": 300}
    assert sel.keep == {300}
    assert "0.060 > 0.05" in sel.ineligible[220]
    assert sel.ineligible[240].startswith("eval failed")


def test_unknown_error_rate_or_missing_metric():
    no_err = done(220, 0.9, 0.9)
    no_err["eval_error"] = None
    only_tr = done(240, 0.5, 0.0)
    del only_tr["metrics"][HF]
    sel = select_keep([220, 240, 260], [], {220: no_err, 240: only_tr, 260: done(260, 0.1, 0.1)}, CFG)
    assert sel.ineligible[220] == "eval error rate unknown"
    assert sel.roles["best_train_reward"] == 240
    assert sel.roles["best_hurdle_frac"] == 260


def test_pending_evals_are_held():
    records = {220: done(220, 0.3, 0.3), 240: status(240, RUNNING)}
    sel = select_keep([220, 240, 260, 280], [], records, CFG)
    # 240 is running, 260 has no record yet, 280 is last (and pending)
    assert sel.keep == {220, 240, 260, 280}
    assert sel.pending == [240, 260, 280]


def test_external_fork_step_can_hold_best_but_is_never_kept_locally():
    records = {200: done(200, 0.24, 0.43), 220: done(220, 0.20, 0.44), 240: done(240, 0.10, 0.10)}
    sel = select_keep([220, 240], [200], records, CFG)
    assert sel.roles == {"best_train_reward": 200, "best_hurdle_frac": 220, "last": 240}
    assert sel.keep == {220, 240}
    assert any("external" in r for r in sel.reasons[200])


def test_external_step_without_a_record_is_not_a_candidate():
    sel = select_keep([220], [200], {220: done(220, 0.1, 0.1)}, CFG)
    assert sel.roles["best_train_reward"] == 220


def test_records_of_incomplete_steps_are_ignored():
    # 260: an eval that finished for a step whose trainer save never completed (lost timeline)
    records = {220: done(220, 0.1, 0.1), 260: done(260, 0.9, 0.9)}
    sel = select_keep([220], [], records, CFG)
    assert sel.roles["best_train_reward"] == 220
    assert 260 not in sel.keep and 260 not in sel.eligible


# ── plan_deletions ─────────────────────────────────────────────────────────


def test_plan_deletions_never_touches_newest_saved_or_above():
    assert plan_deletions([200, 220, 240, 260, 280], keep={240}, newest_saved=260) == [200, 220]
    assert plan_deletions([220, 240], keep=set(), newest_saved=None) == []
    assert plan_deletions([220, 240], keep=set(), newest_saved=300, protected={220}) == [240]


# ── eval_error ─────────────────────────────────────────────────────────────


def test_eval_error_matches_auto_export_rule():
    m = {
        f"eval/{ENV}/all/has_error/mean": 0.10,
        f"eval/{ENV}/all/cancelled/mean": 0.05,
        f"eval/{ENV}/all/dispatch_failure/mean": 0.10,
    }
    assert eval_error(m, ENV) == pytest.approx(1 - 0.85 * 0.90)
    assert eval_error({f"eval/{ENV}/all/agent/has_error/mean": 0.02}, ENV) == pytest.approx(0.02)
    assert eval_error({}, ENV) is None
    assert eval_error({f"eval/{ENV}/all/has_error/mean": 0.9, f"eval/{ENV}/all/cancelled/mean": 0.5}, ENV) == 1.0


# ── EvalRecorder ───────────────────────────────────────────────────────────


def test_recorder_trigger_done_and_rerun(tmp_path):
    rec = EvalRecorder(tmp_path, ckpt_interval=20, now=lambda: 0.0)
    rec.on_trigger(220, fired=[ENV], env_names=[ENV, "other"], expected={ENV: 120}, is_final=False)
    records = read_records(tmp_path, ENV)
    assert records[220]["status"] == RUNNING and records[220]["expected"] == 120
    # "other" did not fire at a checkpoint step: skipped, so it is not held as pending forever
    assert read_records(tmp_path, "other")[220]["status"] == SKIPPED

    metrics = {
        TR: 0.25,
        HF: 0.45,
        f"eval/{ENV}/all/has_error/mean": 0.01,
        f"eval/{ENV}/effective/agent/metrics/train_reward/mean": 0.26,
        f"eval/{ENV}/all/agent/error/X": None,
        f"eval/{ENV}/policy_version": 221.0,
        "step": 220.0,
    }
    out = rec.on_done(ENV, 220, metrics=metrics, n_episodes=120, n_failures=0, n_cancelled=0, policy_version=221)
    assert out["eval_error"] == pytest.approx(0.01)
    saved = read_records(tmp_path, ENV)[220]
    assert saved["status"] == DONE
    assert set(saved["metrics"]) == {TR, HF, f"eval/{ENV}/all/has_error/mean", f"eval/{ENV}/policy_version"}
    assert saved["policy_version"] == 221 and saved["n_episodes"] == 120

    # A re-run of the same step (retrigger on resume) overwrites: last wins
    rec.on_trigger(220, fired=[ENV], env_names=[ENV], expected={ENV: 120}, is_final=False)
    assert read_records(tmp_path, ENV)[220]["status"] == RUNNING
    assert not list(eval_results_dir(tmp_path, ENV).glob("*.tmp"))


def test_recorder_non_checkpoint_steps_and_skipped_versions(tmp_path):
    rec = EvalRecorder(tmp_path, ckpt_interval=20)
    rec.on_trigger(201, fired=[], env_names=[ENV], expected={}, is_final=False)
    assert read_records(tmp_path, ENV) == {}
    # The watcher jumped from 201 to 245: the checkpoint steps in between were never triggered
    rec.on_trigger(245, fired=[], env_names=[ENV], expected={}, is_final=False)
    records = read_records(tmp_path, ENV)
    assert {s: r["status"] for s, r in records.items()} == {220: SKIPPED, 240: SKIPPED}
    # The final step is a checkpoint step whatever the interval
    rec.on_trigger(250, fired=[], env_names=[ENV], expected={}, is_final=True)
    assert read_records(tmp_path, ENV)[250]["status"] == SKIPPED


def test_recorder_failed(tmp_path):
    rec = EvalRecorder(tmp_path, ckpt_interval=20)
    rec.on_failed(ENV, 240, reason="no attempts returned")
    assert read_records(tmp_path, ENV)[240]["status"] == FAILED


def test_reconcile_on_resume(tmp_path):
    for step in (220, 240, 260):
        make_step(tmp_path, step)
    write(tmp_path, done(200, 0.2, 0.4))
    write(tmp_path, status(220, RUNNING))  # interrupted eval below the resume step
    # 240: complete checkpoint without any record
    write(tmp_path, status(260, RUNNING))  # the resume step's own interrupted eval
    write(tmp_path, done(280, 0.9, 0.9))  # lost timeline
    rec = EvalRecorder(tmp_path, ckpt_interval=20)
    lines = rec.reconcile(env_names=[ENV], resume_step=260, complete_steps=[220, 240, 260], retrigger_on_resume=True)
    records = read_records(tmp_path, ENV)
    assert 280 not in records
    assert records[200]["status"] == DONE
    assert records[220]["status"] == LOST
    assert records[240]["status"] == LOST
    assert records[260]["status"] == RUNNING  # re-triggered at startup, which overwrites it
    assert len(lines) == 3

    # Without retrigger_on_resume, the resume step's eval is not re-run
    rec.reconcile(env_names=[ENV], resume_step=260, complete_steps=[220, 240, 260], retrigger_on_resume=False)
    assert read_records(tmp_path, ENV)[260]["status"] == LOST


def test_reconcile_from_scratch_drops_everything(tmp_path):
    write(tmp_path, done(20, 0.1, 0.1))
    EvalRecorder(tmp_path, ckpt_interval=20).reconcile(
        env_names=[ENV], resume_step=None, complete_steps=[], retrigger_on_resume=True
    )
    assert read_records(tmp_path, ENV) == {}


def test_read_records_skips_malformed(tmp_path):
    d = eval_results_dir(tmp_path, ENV)
    d.mkdir(parents=True)
    (d / "step_220.json").write_text("{not json")
    (d / "step_240.json").write_text(json.dumps({"step": 999, "status": DONE}))
    (d / "step_260.json").write_text(json.dumps({"step": 260, "status": "weird"}))
    (d / "notes.txt").write_text("x")
    write(tmp_path, done(280, 0.1, 0.1))
    assert list(read_records(tmp_path, ENV)) == [280]


# ── Keep3Retention (the trainer master's deleter) ──────────────────────────


def test_scan_ckpt_dir_ignores_eval_results(tmp_path):
    make_step(tmp_path, 220)
    make_step(tmp_path, 240, orchestrator=False)
    make_step(tmp_path, 260, trainer=False)
    write(tmp_path, done(220, 0.1, 0.1))
    (tmp_path / RETENTION_FILE).write_text("{}")
    scan = scan_ckpt_dir(tmp_path)
    assert scan.present == (220, 240, 260)
    assert scan.complete == (220,)
    assert scan.trainer_complete == (220, 240)


def test_retention_deletes_only_below_newest_saved(tmp_path):
    ckpt = tmp_path / "run" / "checkpoints"
    external = tmp_path / "other" / "checkpoints" / "step_200"
    make_step(external.parent, 200)
    for step in (220, 240, 260, 280):
        make_step(ckpt, step)
    make_step(ckpt, 300, orchestrator=False)  # final step: orchestrator marker comes after its eval
    make_step(ckpt, 230, trainer=False)  # partial leftover below the newest save
    make_step(ckpt, 320, trainer=False)  # orchestrator checkpoint of a step the trainer has not reached
    write(ckpt, done(200, 0.30, 0.50))
    write(ckpt, done(220, 0.20, 0.60))
    write(ckpt, done(240, 0.10, 0.10))
    write(ckpt, status(260, RUNNING))
    write(ckpt, done(280, 0.10, 0.10))
    external_steps, protected = fork_steps(external, ckpt)
    assert external_steps == {200} and protected == set()
    r = Keep3Retention(ckpt, cfg=CFG, eval_env=ENV, external_steps=external_steps)
    out = r.run(newest_saved=300)
    assert out.selection.roles == {"best_train_reward": 200, "best_hurdle_frac": 220, "last": 280}
    assert out.deleted == [230, 240]
    assert scan_ckpt_dir(ckpt).present == (220, 260, 280, 300, 320)
    assert (external / "trainer" / ".metadata").exists()
    info = json.loads((ckpt / RETENTION_FILE).read_text())
    assert info["keep"] == [220, 260, 280] and info["deleted_now"] == [230, 240]

    # The final prune once the final eval is done: the final step counts as complete (assume_complete), but its
    # orchestrator marker is not written yet, so 280 (what a restart resumes from) stays as the resume point.
    write(ckpt, done(260, 0.05, 0.05))
    write(ckpt, done(300, 0.25, 0.30))
    out = r.run(newest_saved=300, assume_complete=[300])
    assert out.selection.roles == {"best_train_reward": 200, "best_hurdle_frac": 220, "last": 300}
    assert out.deleted == [260]
    assert "resume point" in out.selection.reasons[280][0]
    assert scan_ckpt_dir(ckpt).present == (220, 280, 300, 320)
    # Once the orchestrator writes step 300's checkpoint, 300 is the resume point and 280 goes.
    (ckpt / "step_300" / "orchestrator" / "progress.pt").write_text("p")
    out = r.run(newest_saved=300, assume_complete=[300])
    assert out.deleted == [280]
    assert scan_ckpt_dir(ckpt).present == (220, 300, 320)


def test_retention_without_own_save_deletes_nothing(tmp_path):
    for step in (220, 240):
        make_step(tmp_path, step)
    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV)
    assert r.run(newest_saved=None).deleted == []
    assert scan_ckpt_dir(tmp_path).present == (220, 240)


def test_retention_retries_a_failed_delete(tmp_path):
    for step in (220, 240):
        make_step(tmp_path, step)
    write(tmp_path, done(220, 0.1, 0.1))
    write(tmp_path, done(240, 0.2, 0.2))
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("busy")
        shutil.rmtree(path)

    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV, rmtree=flaky)
    assert r.run(newest_saved=240).failed == {220: "busy"}
    assert r.run(newest_saved=240).deleted == [220]
    assert scan_ckpt_dir(tmp_path).present == (240,)


def test_retention_protects_a_fork_inside_its_own_dir(tmp_path):
    make_step(tmp_path, 200)
    make_step(tmp_path, 220)
    write(tmp_path, done(220, 0.2, 0.2))
    external, protected = fork_steps(tmp_path / "step_200", tmp_path)
    assert external == set() and protected == {200}
    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV, protected_steps=protected)
    write(tmp_path, done(200, 0.1, 0.1))
    assert r.run(newest_saved=220).deleted == []


def test_retention_warns_once_without_eval_writer(tmp_path):
    class Log:
        def __init__(self):
            self.warnings = []

        def warning(self, msg):
            self.warnings.append(msg)

        def info(self, msg):
            pass

    log = Log()
    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV, log=log)
    r.run(newest_saved=None)
    r.run(newest_saved=None)
    assert len(log.warnings) == 1
    EvalRecorder(tmp_path, ckpt_interval=20).write_writer_info({"eval_envs": [ENV]})
    assert (eval_results_dir(tmp_path) / WRITER_FILE).exists()


def test_simulated_run_keeps_at_most_three_plus_pending(tmp_path):
    """Steps 201..300 with checkpoints every 20; each eval finishes 30 steps after its checkpoint."""
    ckpt = tmp_path / "checkpoints"
    rec = EvalRecorder(ckpt, ckpt_interval=20)
    r = Keep3Retention(ckpt, cfg=CFG, eval_env=ENV, external_steps={200})
    scores = {200: (0.24, 0.43), 220: (0.22, 0.45), 240: (0.26, 0.41), 260: (0.20, 0.40), 280: (0.21, 0.39)}
    scores[300] = (0.23, 0.42)
    rec.on_trigger(200, fired=[ENV], env_names=[ENV], expected={ENV: 120}, is_final=False)
    newest = None
    for step in range(201, 301):
        is_final = step == 300
        if step % 20 == 0:
            make_step(ckpt, step, orchestrator=not is_final)
            newest = step
            rec.on_trigger(step, fired=[ENV], env_names=[ENV], expected={ENV: 120}, is_final=is_final)
        finished = step - 30
        if finished in scores and finished % 20 == 0 and finished != 300:
            tr, hf = scores[finished]
            rec.on_done(
                ENV,
                finished,
                metrics={TR: tr, HF: hf, ERR: 0.0},
                n_episodes=1,
                n_failures=0,
                n_cancelled=0,
                policy_version=finished,
            )
        out = r.run(newest)
        local = set(scan_ckpt_dir(ckpt).present)
        # 3 roles at most, plus the pending evals, plus the newest save
        assert len(local - set(out.selection.pending) - {newest}) <= 3
    # End of training: the final eval finishes, the orchestrator writes the final marker, the final prune runs
    for finished in (280, 300):
        tr, hf = scores[finished]
        rec.on_done(
            ENV,
            finished,
            metrics={TR: tr, HF: hf, ERR: 0.0},
            n_episodes=1,
            n_failures=0,
            n_cancelled=0,
            policy_version=finished,
        )
    (ckpt / "step_300" / "orchestrator" / "progress.pt").write_text("p")
    out = r.run(300)
    assert out.selection.roles == {"best_train_reward": 240, "best_hurdle_frac": 220, "last": 300}
    assert scan_ckpt_dir(ckpt).present == (220, 240, 300)


def test_run_safely_logs_instead_of_raising(tmp_path):
    make_step(tmp_path, 220)
    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV)
    r.select = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert r.run_safely(newest_saved=220) is None


def test_recorder_terminal_at(tmp_path):
    rec = EvalRecorder(tmp_path, ckpt_interval=20)
    assert not rec.terminal_at(200, [ENV])
    rec.on_trigger(200, fired=[ENV], env_names=[ENV, "other"], expected={ENV: 320}, is_final=False)
    assert not rec.terminal_at(200, [ENV])  # running
    assert rec.terminal_at(200, ["other"])  # skipped: not due at this step
    rec.on_done(ENV, 200, metrics={TR: 0.2}, n_episodes=320, n_failures=0, n_cancelled=0, policy_version=200)
    assert rec.terminal_at(200, [ENV, "other"])
    assert rec.terminal_at(200, [ENV, "other"], statuses=(DONE, SKIPPED))
    assert not rec.terminal_at(200, [])
    rec.on_failed(ENV, 200, reason="no attempts returned")
    assert rec.terminal_at(200, [ENV]) and not rec.terminal_at(200, [ENV], statuses=(DONE, SKIPPED))


# ── review fixes: the final-eval window, removal order, unreadable records ─


def test_select_keep_keeps_the_resume_point_whatever_its_roles():
    # 300 assumed complete (no orchestrator marker yet); 280 is the newest step a restart can resume from.
    records = {s: done(s, 0.1, 0.1) for s in (240, 280, 300)}
    records[200] = done(200, 0.9, 0.9)
    sel = select_keep([240, 280, 300], [200], records, CFG, resume_point=280)
    assert sel.last == 300 and sel.keep == {280, 300}
    assert sel.roles == {"best_train_reward": 200, "best_hurdle_frac": 200, "last": 300}
    assert "resume point" in sel.reasons[280][0]
    # Without the extra step the resume point is the last one: nothing changes.
    assert select_keep([240, 280], [200], records, CFG, resume_point=280).keep == {280}


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        pass


@pytest.mark.parametrize("local_best", [False, True])
def test_final_eval_window_never_deletes_what_a_restart_resumes_from(tmp_path, local_best):
    """The trainer polls keep-3 with the final step assumed complete while eval@300 runs (hours); a platform restart
    meanwhile must still find step 280 complete: not a crash loop (both bests external) or a rewind to 240."""
    from types import SimpleNamespace

    from prime_rl.utils.pathing import resolve_resume

    ckpt = tmp_path / "run" / "checkpoints"
    rec = EvalRecorder(ckpt, ckpt_interval=20)
    rec.write_writer_info({})
    write(ckpt, done(200, 0.24, 0.43))  # the fork step, external
    make_step(ckpt, 240)
    write(ckpt, done(240, 0.30, 0.50) if local_best else done(240, 0.10, 0.10))
    make_step(ckpt, 280)
    write(ckpt, done(280, 0.20, 0.40))
    make_step(ckpt, 300, orchestrator=False)
    write(ckpt, status(300, RUNNING))
    r = Keep3Retention(ckpt, cfg=CFG, eval_env=ENV, external_steps={200})
    r.run(300)  # after the final save
    out = r.run(300, assume_complete=[300])  # the wait's polls
    assert 280 in out.selection.keep and 280 not in out.deleted
    resume = SimpleNamespace(dir=tmp_path / "il976" / "checkpoints" / "step_200", dir_step=200, step=None)
    assert resolve_resume(resume, ckpt) == (280, None)
    assert scan_ckpt_dir(ckpt).present == ((240, 280, 300) if local_best else (280, 300))


def test_removal_deletes_the_completion_markers_first(tmp_path):
    for step in (220, 240):
        make_step(tmp_path, step)
    write(tmp_path, done(220, 0.1, 0.1))
    write(tmp_path, done(240, 0.2, 0.2))
    (eval_results_dir(tmp_path) / WRITER_FILE).write_text("{}")

    def killed_halfway(path):
        raise OSError("killed mid-removal")

    r = Keep3Retention(tmp_path, cfg=CFG, eval_env=ENV, rmtree=killed_halfway)
    out = r.run(newest_saved=240)
    assert out.failed and 220 not in out.deleted
    # The half-removed step no longer looks complete: no restart resumes from it, and the next pass removes it.
    assert scan_ckpt_dir(tmp_path).present == (220, 240) and scan_ckpt_dir(tmp_path).complete == (240,)
    r.rmtree = shutil.rmtree
    assert r.run(newest_saved=240).deleted == [220]
    assert scan_ckpt_dir(tmp_path).present == (240,)


def test_removal_refuses_a_symlinked_step_dir(tmp_path):
    ckpt = tmp_path / "ckpt"
    elsewhere = tmp_path / "elsewhere" / "step_220"
    make_step(elsewhere.parent, 220)
    ckpt.mkdir()
    (ckpt / "step_220").symlink_to(elsewhere, target_is_directory=True)
    make_step(ckpt, 240)
    write(ckpt, done(220, 0.1, 0.1))
    write(ckpt, done(240, 0.2, 0.2))
    out = Keep3Retention(ckpt, cfg=CFG, eval_env=ENV, log=_Log()).run(newest_saved=240)
    assert 220 in out.failed and out.deleted == []
    assert (elsewhere / "trainer" / ".metadata").exists() and (elsewhere / "orchestrator" / "progress.pt").exists()


def test_reconcile_leaves_an_unreadable_record_alone(tmp_path):
    """A record that exists but cannot be read at startup (a transient FS error) must not become ``lost``: a best
    step would lose its role and be deleted. One above the resume step is still dropped (lost timeline)."""
    for step in (220, 240):
        make_step(tmp_path, step)
    d = eval_results_dir(tmp_path, ENV)
    d.mkdir(parents=True)
    (d / "step_220.json").write_text("{torn")
    (d / "step_260.json").write_text("{torn")
    log = _Log()
    rec = EvalRecorder(tmp_path, ckpt_interval=20, log=log)
    lines = rec.reconcile(env_names=[ENV], resume_step=240, complete_steps=[220, 240], retrigger_on_resume=True)
    assert (d / "step_220.json").read_text() == "{torn"
    assert not (d / "step_260.json").exists()
    assert any("unreadable" in w for w in log.warnings)
    assert lines == [f"dropped {ENV} eval record of step 260 (above resume step 240: lost timeline)"]


def test_reconcile_warns_when_the_fork_steps_eval_is_lost(tmp_path):
    make_step(tmp_path, 220)
    write(tmp_path, status(200, RUNNING))  # eval@200 of the fork, interrupted after 220 was saved
    log = _Log()
    EvalRecorder(tmp_path, ckpt_interval=20, log=log).reconcile(
        env_names=[ENV], resume_step=220, complete_steps=[220], retrigger_on_resume=True, external_steps={200}
    )
    assert read_records(tmp_path, ENV)[200]["status"] == LOST
    assert any("forked step 200" in w for w in log.warnings)
