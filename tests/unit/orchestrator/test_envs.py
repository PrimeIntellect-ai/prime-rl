"""The ``run_rollout`` unwrap seam: an env server's ``Episode`` becomes one ``Rollout``
per trace sharing the episode stamp; episode errors fail every sibling; frozen seats
drop out of training; a fanned episode needs a ``multi_seat`` algorithm on the train
path (eval keeps the policy-played seats)."""

import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.envs import EvalEnv, TrainEnv
from prime_rl.orchestrator.types import Rollout, complete_episodes


class _StubEnvClient:
    """Hands back a canned episode; no env server is spawned."""

    def __init__(self, episode: vf.WireEpisode):
        self.episode = episode

    async def run_rollout(self, **kwargs) -> vf.WireEpisode:
        return self.episode


class _StubSampler:
    def sampling_args(self, args: dict) -> dict:
        return args


class _SingleSeat(Algorithm):
    def __init__(self):
        pass


class _MultiSeat(Algorithm):
    multi_seat = True

    def __init__(self):
        pass


def _task() -> vf.TraceTask:
    return vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0))


def _trace(*, trainable: bool = True, role: str | None = None) -> vf.WireTrace:
    return vf.WireTrace(task=_task(), trainable=trainable, role=role)


def _episode(traces: list[vf.WireTrace], errors: list[vf.Error] | None = None) -> vf.WireEpisode:
    return vf.WireEpisode(env="dummy", task=_task(), errors=errors or [], traces=traces)


def _train_env(episode: vf.WireEpisode, algorithm: Algorithm | None = None) -> TrainEnv:
    env = TrainEnv(TrainEnvConfig(id="dummy"), _StubSampler(), algorithm or _SingleSeat())
    env._env_client = _StubEnvClient(episode)
    return env


def _eval_env(episode: vf.WireEpisode) -> EvalEnv:
    env = EvalEnv(EvalEnvConfig(id="dummy"))
    env._env_client = _StubEnvClient(episode)
    return env


def _run(env: TrainEnv | EvalEnv) -> list[Rollout]:
    return asyncio.run(env.run_rollout(client=None, task_idx=0, model_name="m", cache_salt=None))


def test_single_trace_episode_becomes_one_rollout_with_episode_errors_folded():
    episode = _episode([_trace()], errors=[vf.Error(type="EnvError", message="score hook failed")])
    (rollout,) = _run(_train_env(episode))
    assert isinstance(rollout, Rollout)
    assert rollout.task.data.idx == 0
    assert [e.message for e in rollout.errors] == ["score hook failed"]
    assert rollout.episode_id == episode.id and rollout.episode_rollouts == 1


def test_empty_episode_becomes_error_marker_carrying_the_task():
    episode = _episode([], errors=[vf.Error(type="EnvError", message="rollout hook failed")])
    (rollout,) = _run(_train_env(episode))
    assert rollout.stop_condition == "error"
    assert rollout.task == episode.task
    assert rollout.error is not None and rollout.error.message == "rollout hook failed"
    assert rollout.episode_rollouts == 1


def test_multi_trace_episode_needs_a_multi_seat_algorithm():
    episode = _episode([_trace(role="proposer"), _trace(role="solver")])
    with pytest.raises(RuntimeError, match="hierarchical_grpo"):
        _run(_train_env(episode))
    rollouts = _run(_train_env(episode, algorithm=_MultiSeat()))
    assert [r.role for r in rollouts] == ["proposer", "solver"]
    assert {r.episode_id for r in rollouts} == {episode.id}
    assert all(r.episode_rollouts == 2 for r in rollouts)


def test_frozen_seats_drop_out_of_training():
    episode = _episode([_trace(role="solver"), _trace(trainable=False, role="judge")])
    rollouts = _run(_train_env(episode, algorithm=_MultiSeat()))
    assert [r.role for r in rollouts] == ["solver"]
    # The episode restamps to what ships, so group accounting stays complete.
    assert rollouts[0].episode_rollouts == 1


def test_episode_errors_fail_every_sibling():
    episode = _episode(
        [_trace(role="proposer"), _trace(role="solver")],
        errors=[vf.Error(type="EnvError", message="score hook failed")],
    )
    rollouts = _run(_train_env(episode, algorithm=_MultiSeat()))
    assert len(rollouts) == 2 and all(r.has_error for r in rollouts)


def test_untrainable_sole_trace_train_refuses_eval_accepts():
    episode = _episode([_trace(trainable=False, role="judge")])
    with pytest.raises(RuntimeError, match="no trainable"):
        _run(_train_env(episode))
    (rollout,) = _run(_eval_env(episode))
    assert rollout.trainable is False


def test_eval_keeps_only_policy_played_seats_of_a_fanned_episode():
    episode = _episode([_trace(role="solver"), _trace(role="solver"), _trace(trainable=False, role="judge")])
    rollouts = _run(_eval_env(episode))
    assert [r.role for r in rollouts] == ["solver", "solver"]
    assert all(r.episode_rollouts == 2 for r in rollouts)


def test_complete_episodes_counts_whole_episodes():
    episode = _episode([_trace(role="a"), _trace(role="b")])
    pair = _run(_train_env(episode, algorithm=_MultiSeat()))
    singleton = Rollout(task=_task())
    assert complete_episodes([pair[0]]) == 0  # half an episode is not a unit
    assert complete_episodes(pair) == 1
    assert complete_episodes([singleton, *pair]) == 2
