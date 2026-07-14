"""Dispatcher client routing: eval groups use the chat-relay eval client, EXCEPT
TTT-enabled eval envs, which must sample through the renderer (train) client — TTT
consumes exact token ids, so the relay would refuse at the first compaction."""

from dataclasses import dataclass

import pytest

pytest.importorskip("verifiers")

from prime_rl.orchestrator.dispatcher import _wants_train_client  # noqa: E402
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


def group(kind: str, env_name: str) -> GroupState:
    return GroupState(kind=kind, env_name=env_name, task_idx=0, rollouts_to_schedule=1, target_rollouts=1)


def test_ttt_eval_env_routes_to_train_client():
    envs = FakeEnvs({"ttt-eval": FakeEnv(FakeConfig(ttt=FakeTTT(enabled=True)))})
    assert _wants_train_client(envs, group("eval", "ttt-eval")) is True


def test_plain_eval_env_keeps_eval_client():
    envs = FakeEnvs(
        {
            "plain": FakeEnv(FakeConfig(ttt=None)),
            "disabled": FakeEnv(FakeConfig(ttt=FakeTTT(enabled=False))),
        }
    )
    assert _wants_train_client(envs, group("eval", "plain")) is False
    assert _wants_train_client(envs, group("eval", "disabled")) is False


def test_eval_client_renderer_routes_all_evals_through_train_client():
    envs = FakeEnvs({"plain": FakeEnv(FakeConfig(ttt=None))})
    assert _wants_train_client(envs, group("eval", "plain"), eval_client="renderer") is True
    assert _wants_train_client(None, group("eval", "plain"), eval_client="renderer") is True


def test_train_groups_always_use_train_client():
    assert _wants_train_client(None, group("train", "any")) is True
