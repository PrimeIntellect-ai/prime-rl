"""The ``run_rollout`` unwrap seam: an env server's ``Episode`` becomes exactly one
``Rollout`` (episode errors folded in), an empty episode becomes an error marker, and
anything not trainable — multi-trace episodes, a frozen sole seat — is refused on the
train path (eval still accepts untrainable traces)."""

import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.envs import EvalEnv, TrainEnv
from prime_rl.orchestrator.types import Rollout


class _StubEnvClient:
    """Hands back a canned episode; no env server is spawned."""

    def __init__(self, episode: vf.WireEpisode):
        self.episode = episode

    async def run_rollout(self, **kwargs) -> vf.WireEpisode:
        return self.episode


class _StubSampler:
    def sampling_args(self, args: dict) -> dict:
        return args


def _task() -> vf.TraceTask:
    return vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0))


def _trace(*, trainable: bool = True, role: str | None = None) -> vf.WireTrace:
    return vf.WireTrace(task=_task(), trainable=trainable, role=role)


def _episode(traces: list[vf.WireTrace], errors: list[vf.Error] | None = None) -> vf.WireEpisode:
    return vf.WireEpisode(env="dummy", task=_task(), errors=errors or [], traces=traces)


def _train_env(episode: vf.WireEpisode) -> TrainEnv:
    env = TrainEnv(TrainEnvConfig(id="dummy"), _StubSampler(), None)
    env._env_client = _StubEnvClient(episode)
    return env


def _eval_env(episode: vf.WireEpisode) -> EvalEnv:
    env = EvalEnv(EvalEnvConfig(id="dummy"))
    env._env_client = _StubEnvClient(episode)
    return env


def _run(env: TrainEnv | EvalEnv) -> Rollout:
    return asyncio.run(env.run_rollout(client=None, task_idx=0, model_name="m", cache_salt=None))


def test_single_trace_episode_becomes_rollout_with_episode_errors_folded():
    episode = _episode([_trace()], errors=[vf.Error(type="EnvError", message="score hook failed")])
    rollout = _run(_train_env(episode))
    assert isinstance(rollout, Rollout)
    assert rollout.task.data.idx == 0
    assert [e.message for e in rollout.errors] == ["score hook failed"]


def test_empty_episode_becomes_error_marker_carrying_the_task():
    episode = _episode([], errors=[vf.Error(type="EnvError", message="rollout hook failed")])
    rollout = _run(_train_env(episode))
    assert rollout.stop_condition == "error"
    assert rollout.task == episode.task
    assert rollout.error is not None and rollout.error.message == "rollout hook failed"


def test_multi_trace_episode_is_refused():
    episode = _episode([_trace(), _trace(role="judge")])
    with pytest.raises(RuntimeError, match="multi-trace"):
        _run(_train_env(episode))


def test_untrainable_sole_trace_train_refuses_eval_accepts():
    episode = _episode([_trace(trainable=False, role="judge")])
    with pytest.raises(RuntimeError, match="untrainable"):
        _run(_train_env(episode))
    rollout = _run(_eval_env(episode))
    assert rollout.trainable is False
