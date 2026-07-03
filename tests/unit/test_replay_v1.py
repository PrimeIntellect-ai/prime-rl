"""Unit tests for the replay derivation packages (replay-continue-v1 / replay-recheck-v1):
config validation and anchor/prompt binding. The shared base's surgery and buffer logic
is tested in verifiers (tests/v1/test_replay.py)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from replay_continue_v1.taskset import ReplayContinueConfig, ReplayContinueTaskset
from replay_recheck_v1.taskset import ReplayRecheckConfig
from reverse_text_v1.taskset import ReverseTextConfig
from verifiers.v1.loaders import task_type, taskset_class, taskset_config_type

INNER = {"id": "reverse-text-v1"}


def test_loader_resolves_both_packages():
    assert taskset_class("replay-continue-v1") is ReplayContinueTaskset
    assert taskset_config_type("replay-continue-v1") is ReplayContinueConfig
    assert taskset_config_type("replay-recheck-v1") is ReplayRecheckConfig
    assert task_type("replay-recheck-v1").__name__ == "ReplayTask"


def test_config_requires_inner():
    # `inner` is a required base field, so leaving it out is a plain missing-field error.
    with pytest.raises(ValidationError, match="inner"):
        ReplayRecheckConfig(buffer_dir="/tmp/buf")


def test_config_rejects_unknown_anchor():
    with pytest.raises(ValidationError, match="anchor"):
        ReplayContinueConfig(buffer_dir="/tmp/buf", anchor="turn", inner=INNER)


def test_config_requires_buffer_dir():
    with pytest.raises(ValidationError, match="buffer_dir"):
        ReplayRecheckConfig(inner=INNER)


def test_config_self_buffer_implies_online():
    cfg = ReplayRecheckConfig(buffer_dir="self", inner=INNER)
    assert cfg.online is True  # pinned before the orchestrator rewrites the sentinel


def test_config_narrows_inner():
    cfg = ReplayContinueConfig(buffer_dir="/tmp/buf", anchor="tool-call", inner=INNER)
    assert isinstance(cfg.inner, ReverseTextConfig)  # narrowed to the inner taskset's type


def _node(parent: int | None, sampled: bool, message: dict) -> dict:
    return {"parent": parent, "sampled": sampled, "message": message}


@pytest.fixture
def tool_buffer(tmp_path: Path) -> Path:
    call = {"name": "t", "arguments": "{}"}
    record = {
        "id": "ttt",
        "errors": None,
        "stop_condition": "agent_completed",
        "rewards": {"r": 1.0},
        "task": {"idx": 0, "prompt": "Reverse: abc", "answer": "cba"},
        "info": {},
        "nodes": [
            _node(None, False, {"role": "system", "content": "s"}),
            _node(0, False, {"role": "user", "content": "task"}),
            _node(1, True, {"role": "assistant", "content": "", "tool_calls": [call | {"id": "a"}]}),
            _node(2, False, {"role": "tool", "tool_call_id": "a", "content": "ra"}),
            _node(3, True, {"role": "assistant", "content": "", "tool_calls": [call | {"id": "b"}]}),
            _node(4, False, {"role": "tool", "tool_call_id": "b", "content": "rb"}),
            _node(5, True, {"role": "assistant", "content": "done"}),
        ],
    }
    step = tmp_path / "step_1"
    step.mkdir()
    (step / "train_rollouts.jsonl").write_text(json.dumps(record) + "\n")
    return tmp_path


def test_continue_tool_call_anchor_is_sampled_deterministically(tool_buffer):
    def scan(taskset):
        return taskset.buffer.scan()

    make = lambda: ReplayContinueTaskset(  # noqa: E731
        ReplayContinueConfig(buffer_dir=str(tool_buffer), anchor="tool-call", inner=INNER)
    )
    first, second = scan(make()), scan(make())
    # One resume point per source rollout, identical across taskset instances (pool workers).
    assert len(first) == 1 and first[0].anchor_node in (3, 5)
    assert first[0].anchor_node == second[0].anchor_node
    # Compaction anchoring finds nothing in this rollout (it never compacted).
    compaction = ReplayContinueTaskset(ReplayContinueConfig(buffer_dir=str(tool_buffer), inner=INNER))
    assert compaction.buffer.scan() == []
