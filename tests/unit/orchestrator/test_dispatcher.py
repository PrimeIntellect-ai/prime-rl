"""Dispatcher client routing: eval groups use the chat-relay eval client, EXCEPT
TTT-enabled eval envs, which must sample through the renderer (train) client — TTT
consumes exact token ids, so the relay would refuse at the first compaction."""

from dataclasses import dataclass

import pytest

pytest.importorskip("verifiers")

from prime_rl.orchestrator.dispatcher import RolloutDispatcher  # noqa: E402
from prime_rl.orchestrator.types import GroupState  # noqa: E402


@dataclass
class FakeTTT:
    enabled: bool = True


@dataclass
class FakeConfig:
    ttt: FakeTTT | None = None


@dataclass
class FakeEnv:
    config: FakeConfig


class FakeEnvs:
    def __init__(self, envs: dict[str, FakeEnv]):
        self._envs = envs

    def get(self, name: str) -> FakeEnv:
        return self._envs[name]


def make_dispatcher(eval_envs: FakeEnvs | None) -> RolloutDispatcher:
    # ``_wants_train_client`` only reads ``eval_envs``; skip the heavy constructor.
    dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
    dispatcher.eval_envs = eval_envs
    return dispatcher


def group(kind: str, env_name: str) -> GroupState:
    return GroupState(kind=kind, env_name=env_name, task_idx=0, rollouts_to_schedule=1, target_rollouts=1)


def test_ttt_eval_env_routes_to_train_client():
    envs = FakeEnvs({"ttt-eval": FakeEnv(FakeConfig(ttt=FakeTTT(enabled=True)))})
    assert make_dispatcher(envs)._wants_train_client(group("eval", "ttt-eval")) is True


def test_plain_eval_env_keeps_eval_client():
    envs = FakeEnvs(
        {
            "plain": FakeEnv(FakeConfig(ttt=None)),
            "disabled": FakeEnv(FakeConfig(ttt=FakeTTT(enabled=False))),
        }
    )
    dispatcher = make_dispatcher(envs)
    assert dispatcher._wants_train_client(group("eval", "plain")) is False
    assert dispatcher._wants_train_client(group("eval", "disabled")) is False


def test_train_groups_always_use_train_client():
    assert make_dispatcher(None)._wants_train_client(group("train", "any")) is True
