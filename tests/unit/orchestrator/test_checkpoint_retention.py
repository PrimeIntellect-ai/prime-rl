import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.ckpt import (
    CheckpointRetention,
    eval_epoch_score,
    eval_plan_fingerprint,
    select_best_eval,
)
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import EvalBatch
from prime_rl.utils.pathing import (
    complete_joint_ckpt_step,
    get_generation_path,
    get_joint_ckpt_steps,
    get_resume_ckpt_steps,
    get_retention_path,
    get_step_path,
    is_joint_ckpt_step,
    preflight_checkpoint_storage,
    prune_joint_ckpts,
    require_joint_resume_step,
    start_checkpoint_generation,
)


def _components(ckpt_dir, step, *, trainer=True, orchestrator=True):
    step_path = get_step_path(ckpt_dir, step)
    if trainer:
        (step_path / "trainer").mkdir(parents=True, exist_ok=True)
        (step_path / "trainer" / ".metadata").write_bytes(b"dcp")
    if orchestrator:
        (step_path / "orchestrator").mkdir(parents=True, exist_ok=True)
        (step_path / "orchestrator" / "progress.pt").write_bytes(b"progress")


def _joint(ckpt_dir, step, *, generation="g1"):
    _components(ckpt_dir, step)
    if not get_generation_path(ckpt_dir).exists():
        start_checkpoint_generation(ckpt_dir, generation=generation, resume_step=None)
    complete_joint_ckpt_step(ckpt_dir, step, component="trainer", generation=generation)
    assert complete_joint_ckpt_step(ckpt_dir, step, component="orchestrator", generation=generation)


def test_joint_discovery_and_resume_skip_partial_and_directory_only(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    get_step_path(ckpt_dir, 20).mkdir(parents=True)
    _components(ckpt_dir, 40)
    _joint(ckpt_dir, 60)
    _components(ckpt_dir, 80, trainer=False)
    assert not complete_joint_ckpt_step(ckpt_dir, 80, component="orchestrator", generation="g1")

    assert get_joint_ckpt_steps(ckpt_dir) == [60]
    assert require_joint_resume_step(ckpt_dir) == 60
    assert require_joint_resume_step(ckpt_dir, 60) == 60
    with pytest.raises(FileNotFoundError, match="not jointly complete"):
        require_joint_resume_step(ckpt_dir, 40)
    with pytest.raises(FileNotFoundError, match="No jointly complete"):
        require_joint_resume_step(tmp_path / "empty")


def test_pruning_keeps_latest_best_pending_and_partial(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for step in (20, 40, 60, 80):
        _joint(ckpt_dir, step)
    _components(ckpt_dir, 100)  # not a cleanup candidate until jointly committed
    get_retention_path(ckpt_dir).write_text(
        json.dumps({"best_step": 20, "pending_eval": {"40": ["heldout"]}, "valid_eval": {}})
    )

    assert prune_joint_ckpts(ckpt_dir, generation="g1", keep_last=1, keep_interval=None) == [60]
    assert get_joint_ckpt_steps(ckpt_dir) == [20, 40, 80]
    assert get_step_path(ckpt_dir, 100).exists()


def test_failed_rmtree_stays_in_the_next_cleanup_plan(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    for step in (20, 60):
        _joint(ckpt_dir, step)
    import prime_rl.utils.pathing as pathing

    real_rmtree = pathing.shutil.rmtree
    failed_steps = set()

    def partially_fail(path, *args, **kwargs):
        # Even losing the manifest must not drop this path from the retry plan.
        (Path(path) / "manifest.json").unlink()
        raise OSError("busy")

    monkeypatch.setattr(pathing.shutil, "rmtree", partially_fail)
    assert (
        prune_joint_ckpts(ckpt_dir, generation="g1", keep_last=1, keep_interval=None, failed_steps=failed_steps) == []
    )
    assert failed_steps == {20}
    assert get_joint_ckpt_steps(ckpt_dir) == [60]
    monkeypatch.setattr(pathing.shutil, "rmtree", real_rmtree)
    assert prune_joint_ckpts(ckpt_dir, generation="g1", keep_last=1, keep_interval=None, failed_steps=failed_steps) == [
        20
    ]
    assert failed_steps == set()
    assert get_joint_ckpt_steps(ckpt_dir) == [60]


def test_storage_preflight_checks_transient_space_and_cleans_probe(tmp_path, monkeypatch):
    import prime_rl.utils.pathing as pathing

    monkeypatch.setattr(pathing.shutil, "disk_usage", lambda _: SimpleNamespace(free=99))
    with pytest.raises(OSError, match="Insufficient checkpoint storage"):
        preflight_checkpoint_storage(tmp_path / "checkpoints", required_bytes=100)
    preflight_checkpoint_storage(tmp_path / "checkpoints", required_bytes=99)
    assert list((tmp_path / "checkpoints").glob(".storage-preflight.*")) == []


def _episode(task_key, step, reward, *, ok=True, start=None, end=None, named_reward=True):
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None), key=task_key, hash=task_key)
    trace = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        rewards={"continuous_reward": vf.Reward(score=reward)} if named_reward else {},
        ok=ok,
        is_completed=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id="heldout", name="heldout"),
        task=task,
        group=vf.GroupInfo(id=f"group-{task_key}"),
        run=vf.TrainRunInfo(
            id="run",
            work=vf.EvalWorkInfo(
                step=step,
                policy=vf.PolicySpan(start=step if start is None else start, end=step if end is None else end),
            ),
        ),
        traces=[trace],
        ok=ok,
    )


def _batch(step=40, episodes=None, *, failures=None, cancelled=0):
    episodes = (
        episodes
        if episodes is not None
        else [
            _episode("a", step, 0.25),
            _episode("a", step, 0.75),
            _episode("b", step, 0.5),
            _episode("b", step, 1.0),
        ]
    )
    return EvalBatch(
        env_name="heldout",
        step=step,
        episodes=EvalEpisodes(episodes, group_size=2),
        failures=failures or [],
        cancelled=cancelled,
    )


def test_best_eligibility_requires_complete_valid_fixed_policy_census():
    assert eval_epoch_score(_batch(), expected_tasks=2, group_size=2) == pytest.approx(0.625)
    assert eval_epoch_score(_batch(episodes=_batch().episodes.episodes[:3]), expected_tasks=2, group_size=2) is None
    assert eval_epoch_score(_batch(failures=[object()]), expected_tasks=2, group_size=2) is None
    assert eval_epoch_score(_batch(cancelled=1), expected_tasks=2, group_size=2) is None
    assert (
        eval_epoch_score(
            _batch(episodes=[_episode("a", 40, 1.0, end=41), *_batch().episodes.episodes[1:]]),
            expected_tasks=2,
            group_size=2,
        )
        is None
    )
    assert (
        eval_epoch_score(
            _batch(episodes=[_episode("a", 40, 1.0, ok=False), *_batch().episodes.episodes[1:]]),
            expected_tasks=2,
            group_size=2,
        )
        is None
    )
    assert (
        eval_epoch_score(
            _batch(episodes=[_episode("a", 40, 1.0, named_reward=False), *_batch().episodes.episodes[1:]]),
            expected_tasks=2,
            group_size=2,
        )
        is None
    )
    unequal_groups = [_episode("a", 40, 1.0) for _ in range(3)] + [_episode("b", 40, 1.0)]
    assert eval_epoch_score(_batch(episodes=unequal_groups), expected_tasks=2, group_size=2) is None


def test_best_selection_requires_all_valid_envs_and_joint_checkpoint():
    scores = {("heldout", 20): 0.5, ("heldout", 40): 0.75, ("heldout", 60): 0.75}
    assert select_best_eval(scores, joint_steps={20, 40}, required_envs={"heldout"}) == (40, 0.75)
    assert select_best_eval(scores, joint_steps={20, 40, 60}, required_envs={"heldout"}) == (40, 0.75)
    assert select_best_eval(scores, joint_steps=set(), required_envs={"heldout"}) is None
    assert select_best_eval(scores, joint_steps={20, 40, 60}, required_envs={"heldout", "other"}) is None
    assert select_best_eval({}, joint_steps={20}, required_envs={"heldout"}) is None


def test_retention_dedupes_valid_epochs_and_pins_until_joint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _joint(ckpt_dir, 40)
    retention = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
    retention.configure_eval({"heldout": "plan6"})
    retention.mark_eval_pending(40, ["heldout"])
    retention.complete_eval_epoch(_batch(), score=0.625)

    assert retention.valid_eval_envs(40) == {"heldout"}
    assert retention.best_step == 40
    assert retention.state["pending_eval"] == {}
    resumed = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
    resumed.configure_eval({"heldout": "plan6"})
    assert resumed.valid_eval_envs(40) == {"heldout"}
    assert resumed.best_step == 40

    env = SimpleNamespace(
        name="heldout",
        examples=[SimpleNamespace(key="a", hash="a"), SimpleNamespace(key="b", hash="b")],
        config=SimpleNamespace(group_size=2),
    )
    source = EvalSource([env])
    fired, _ = source.trigger(40, skip_envs=resumed.valid_eval_envs(40))
    assert fired == [] and not source.queue

    retention.mark_eval_pending(60, ["heldout"])
    retention.complete_eval_epoch(_batch(step=60), score=None)
    assert retention.valid_eval_envs(60) == set()
    assert retention.state["pending_eval"] == {}
    assert retention.best_step == 40

    _components(ckpt_dir, 80)
    retention.mark_eval_pending(80, ["heldout"])
    retention.complete_eval_epoch(_batch(step=80), score=0.9)
    assert retention.best_step == 40
    assert retention.state["pending_eval"] == {"80": ["heldout"]}
    _joint(ckpt_dir, 80)
    retention.reconcile()
    assert retention.best_step == 80
    assert retention.state["pending_eval"] == {}


def test_rewind_ceiling_excludes_abandoned_futures_and_mixed_generations(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for step in (20, 40, 60):
        _joint(ckpt_dir, step, generation="gen-a")
    start_checkpoint_generation(ckpt_dir, generation="gen-b", resume_step=20)

    assert get_resume_ckpt_steps(ckpt_dir) == [20]
    assert require_joint_resume_step(ckpt_dir) == 20
    # A new trainer save beside an old orchestrator receipt is not a joint checkpoint.
    assert not complete_joint_ckpt_step(ckpt_dir, 40, component="trainer", generation="gen-b")
    assert not is_joint_ckpt_step(ckpt_dir, 40)
    assert require_joint_resume_step(ckpt_dir) == 20
    assert complete_joint_ckpt_step(ckpt_dir, 40, component="orchestrator", generation="gen-b")
    assert get_resume_ckpt_steps(ckpt_dir) == [20, 40]

    # Neither the active nor a stale generation may prune A's abandoned future.
    assert prune_joint_ckpts(ckpt_dir, generation="gen-a", keep_last=1, keep_interval=None) == []
    assert prune_joint_ckpts(ckpt_dir, generation="gen-b", keep_last=1, keep_interval=None) == []
    assert get_step_path(ckpt_dir, 60).exists()
    start_checkpoint_generation(ckpt_dir, generation="gen-c", resume_step=20)
    assert require_joint_resume_step(ckpt_dir) == 20


def test_external_resume_source_is_never_mutated(tmp_path):
    external = tmp_path / "external" / "checkpoints"
    _joint(external, 60, generation="gen-a")
    before = {path.relative_to(external): path.read_bytes() for path in external.rglob("*") if path.is_file()}

    assert require_joint_resume_step(external, 60) == 60
    start_checkpoint_generation(tmp_path / "current" / "checkpoints", generation="gen-b", resume_step=60)

    after = {path.relative_to(external): path.read_bytes() for path in external.rglob("*") if path.is_file()}
    assert after == before


def test_final_best_and_latest_survive_interval_only_pruning(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for step in (20, 40, 60):
        _joint(ckpt_dir, step)
    get_retention_path(ckpt_dir).write_text(
        json.dumps({"best_step": 40, "best_score": 0.5, "pending_eval": {}, "valid_eval": {}})
    )

    assert prune_joint_ckpts(ckpt_dir, generation="g1", keep_last=None, keep_interval=100) == [20]
    assert get_joint_ckpt_steps(ckpt_dir) == [40, 60]


def test_eval_dedupe_and_best_require_generation_and_plan_fingerprint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _joint(ckpt_dir, 40)
    retention = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
    retention.configure_eval({"heldout": "plan-6"})
    retention.complete_eval_epoch(_batch(), score=0.625)
    assert retention.valid_eval_envs(40) == {"heldout"}
    assert retention.best_step == 40

    retention.configure_eval({"heldout": "plan-3"})
    assert retention.valid_eval_envs(40) == set()
    assert retention.best_step is None
    retention.configure_eval({"heldout": "plan-6"})
    assert retention.valid_eval_envs(40) == {"heldout"}
    assert retention.best_step == 40

    start_checkpoint_generation(ckpt_dir, generation="gen-b", resume_step=40)
    next_generation = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
    next_generation.configure_eval({"heldout": "plan-6"})
    assert next_generation.valid_eval_envs(40) == set()
    assert next_generation.best_step is None


def test_eval_fingerprint_covers_group_size_task_identity_and_resolved_config():
    class Config:
        def __init__(self, group_size, revision="r1", reward="continuous_reward"):
            self.values = {
                "group_size": group_size,
                "env": {"taskset": {"id": "heldout", "revision": revision}},
                "reward": {"name": reward},
            }

        def model_dump(self, *, mode):
            assert mode == "json"
            return self.values

    def env(group_size=6, revision="r1", reward="continuous_reward", task_hash="hash-a"):
        return SimpleNamespace(
            config=Config(group_size, revision, reward),
            examples=[SimpleNamespace(key="task-a", hash=task_hash)],
        )

    baseline = eval_plan_fingerprint(env())
    assert eval_plan_fingerprint(env()) == baseline
    assert eval_plan_fingerprint(env(group_size=3)) != baseline
    assert eval_plan_fingerprint(env(task_hash="hash-b")) != baseline
    assert eval_plan_fingerprint(env(revision="r2")) != baseline
    assert eval_plan_fingerprint(env(reward="other_reward")) != baseline


def test_retention_update_and_prune_are_serialized(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    ckpt_dir = tmp_path / "checkpoints"
    for step in (20, 40):
        _joint(ckpt_dir, step)
    retention = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
    retention.configure_eval({"heldout": "plan-6"})
    writing = Event()
    release_write = Event()
    prune_done = Event()
    original_write = retention._write

    def slow_write():
        writing.set()
        assert release_write.wait(5)
        original_write()

    monkeypatch.setattr(retention, "_write", slow_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(retention.mark_eval_pending, 20, ["heldout"])
        assert writing.wait(2)

        def prune():
            result = prune_joint_ckpts(ckpt_dir, generation="g1", keep_last=1, keep_interval=None)
            prune_done.set()
            return result

        cleanup = pool.submit(prune)
        assert not prune_done.wait(0.05)
        release_write.set()
        pending.result()
        assert cleanup.result() == []

    assert get_step_path(ckpt_dir, 20).exists()


def test_legacy_resume_is_explicit_and_validates_both_components(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _components(ckpt_dir, 40)
    with pytest.raises(FileNotFoundError, match="not jointly complete"):
        require_joint_resume_step(ckpt_dir, 40)
    with pytest.raises(ValueError, match="explicit checkpoint step"):
        require_joint_resume_step(ckpt_dir, allow_legacy=True)
    assert require_joint_resume_step(ckpt_dir, 40, allow_legacy=True) == 40

    (ckpt_dir / "step_40" / "manifest.json").write_text(json.dumps({"complete": True, "step": 40}))
    assert require_joint_resume_step(ckpt_dir, 40, allow_legacy=True) == 40
    (ckpt_dir / "step_40" / "manifest.json").write_text(json.dumps({"complete": True, "step": 40, "generation": "mixed"}))
    with pytest.raises(ValueError, match="refusing legacy fallback"):
        require_joint_resume_step(ckpt_dir, 40, allow_legacy=True)


def test_shared_checkpoint_retention_config_propagates_to_both_components():
    from prime_rl.configs.rl import RLConfig

    config = RLConfig.model_validate(
        {
            "ckpt": {"interval": 20, "keep_last": 1},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.ckpt is not None and config.orchestrator.ckpt is not None
    assert config.trainer.ckpt.interval == config.orchestrator.ckpt.interval == 20
    assert config.trainer.ckpt.keep_last == config.orchestrator.ckpt.keep_last == 1
    assert config.orchestrator.ckpt.export is False

    exporting = RLConfig.model_validate(
        {
            "ckpt": {"interval": 20, "keep_last": 1, "export": True},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert exporting.orchestrator.ckpt is not None and exporting.orchestrator.ckpt.export is True

    trainer_only = RLConfig.model_validate(
        {
            "trainer": {"ckpt": {"interval": 20, "keep_last": 1}},
            "orchestrator": {"renderer": {"name": "default"}, "ckpt": {"interval": 20}},
        }
    )
    assert trainer_only.orchestrator.ckpt is not None
    assert trainer_only.orchestrator.ckpt.keep_last == 1


def test_split_checkpoint_root_is_rejected_before_writers_and_legacy_configs_do_not_export():
    from pydantic import ValidationError

    from prime_rl.configs.rl import RLConfig
    from prime_rl.configs.shared import ResumeConfig

    with pytest.raises(ValidationError, match="ckpt.output_dir.*not supported"):
        RLConfig.model_validate(
            {
                "ckpt": {"interval": 20, "output_dir": "/tmp/split-checkpoints"},
                "trainer": {},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )

    legacy = RLConfig.model_validate(
        {
            "ckpt": {"interval": 20},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert legacy.orchestrator.ckpt is not None and legacy.orchestrator.ckpt.export is False
    with pytest.raises(ValidationError, match="explicit resume.step"):
        ResumeConfig(legacy=True)
    assert ResumeConfig(legacy=True, step=40).legacy is True
