import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.ckpt import CheckpointRetention, eval_epoch_score, select_best_eval
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import EvalBatch
from prime_rl.utils.pathing import (
    complete_joint_ckpt_step,
    get_joint_ckpt_steps,
    get_retention_path,
    get_step_path,
    preflight_checkpoint_storage,
    prune_joint_ckpts,
    require_joint_resume_step,
)


def _components(ckpt_dir, step, *, trainer=True, orchestrator=True):
    step_path = get_step_path(ckpt_dir, step)
    if trainer:
        (step_path / "trainer").mkdir(parents=True, exist_ok=True)
        (step_path / "trainer" / ".metadata").write_bytes(b"dcp")
    if orchestrator:
        (step_path / "orchestrator").mkdir(parents=True, exist_ok=True)
        (step_path / "orchestrator" / "progress.pt").write_bytes(b"progress")


def _joint(ckpt_dir, step):
    _components(ckpt_dir, step)
    assert complete_joint_ckpt_step(ckpt_dir, step)


def test_joint_discovery_and_resume_skip_partial_and_directory_only(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    get_step_path(ckpt_dir, 20).mkdir(parents=True)
    _components(ckpt_dir, 40)
    _joint(ckpt_dir, 60)
    _components(ckpt_dir, 80, trainer=False)
    complete_joint_ckpt_step(ckpt_dir, 80)

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

    assert prune_joint_ckpts(ckpt_dir, keep_last=1, keep_interval=None) == [60]
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
    assert prune_joint_ckpts(ckpt_dir, keep_last=1, keep_interval=None, failed_steps=failed_steps) == []
    assert failed_steps == {20}
    assert get_joint_ckpt_steps(ckpt_dir) == [60]
    monkeypatch.setattr(pathing.shutil, "rmtree", real_rmtree)
    assert prune_joint_ckpts(ckpt_dir, keep_last=1, keep_interval=None, failed_steps=failed_steps) == [20]
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
    retention.mark_eval_pending(40, ["heldout"])
    retention.complete_eval_epoch(_batch(), score=0.625)

    assert retention.valid_eval_envs(40) == {"heldout"}
    assert retention.best_step == 40
    assert retention.state["pending_eval"] == {}
    resumed = CheckpointRetention(ckpt_dir, eval_envs=["heldout"])
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
    complete_joint_ckpt_step(ckpt_dir, 80)
    retention.reconcile()
    assert retention.best_step == 80
    assert retention.state["pending_eval"] == {}


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

    trainer_only = RLConfig.model_validate(
        {
            "trainer": {"ckpt": {"interval": 20, "keep_last": 1}},
            "orchestrator": {"renderer": {"name": "default"}, "ckpt": {"interval": 20}},
        }
    )
    assert trainer_only.orchestrator.ckpt is not None
    assert trainer_only.orchestrator.ckpt.keep_last == 1
