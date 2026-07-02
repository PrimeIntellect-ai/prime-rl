"""Unit tests for the replay-v1 environment's pure logic: message-graph surgery,
verdict parsing, config validation, and the buffer's scan/pick/read plumbing."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from replay_v1.buffer import ReplayBuffer
from replay_v1.surgery import (
    _render_message,
    build_children,
    compaction_forks,
    continue_seed,
    final_leaf,
    main_tree,
    recheck_seed,
    render_transcript,
    unwrap_source_task,
    usable,
)
from replay_v1.taskset import ReplayTasksetConfig, parse_verdict
from reverse_text_v1.taskset import ReverseTextConfig

# ------------------------------------------------------------------ surgery


def _node(parent: int | None, sampled: bool, message: dict) -> dict:
    return {"parent": parent, "sampled": sampled, "message": message}


@pytest.fixture
def forest() -> list[dict]:
    """A synthetic message forest exercising every structure surgery must tell apart.

    Main tree (root 0): a compaction fork off the system root (node 6), a retried-assistant
    twin fork (nodes 2/3), a duplicated tool-result fork (nodes 4/5), and a final assistant
    truncated mid-tool-call (node 9, empty content). Subagent tree (root 10): its own
    compaction-shaped fork (node 12) and the forest's highest leaf index (node 13).
    """
    return [
        _node(None, False, {"role": "system", "content": "You are an agent."}),  # 0
        _node(0, False, {"role": "user", "content": "Original task: sort the list."}),  # 1
        _node(1, True, {"role": "assistant", "content": "first attempt"}),  # 2
        _node(1, True, {"role": "assistant", "content": "retried attempt"}),  # 3: retry twin fork
        _node(3, False, {"role": "tool", "content": "tool output"}),  # 4
        _node(3, False, {"role": "tool", "content": "tool output"}),  # 5: duplicated tool result
        _node(0, False, {"role": "user", "content": "Summary: the agent was sorting a list."}),  # 6: compaction
        _node(
            6,
            True,
            {
                "role": "assistant",
                "content": "resuming work",
                "tool_calls": [{"id": "c1", "name": "search", "arguments": '{"q": "sort"}'}],
            },
        ),  # 7
        _node(7, False, {"role": "tool", "content": "x" * 300}),  # 8
        _node(
            8,
            True,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c2", "name": "search", "arguments": '{"q": "again"}'}],
            },
        ),  # 9: final main-tree leaf, truncated mid-tool-call
        _node(None, False, {"role": "user", "content": "subagent task"}),  # 10: subagent root
        _node(10, True, {"role": "assistant", "content": "subagent answer"}),  # 11
        _node(10, False, {"role": "user", "content": "subagent summary"}),  # 12: compaction-shaped, outside main
        _node(12, True, {"role": "assistant", "content": "subagent resumed"}),  # 13: global max leaf
    ]


def test_main_tree_excludes_subagent_roots(forest):
    children, roots = build_children(forest)
    assert roots == [0, 10]
    assert main_tree(children) == set(range(10))


def test_compaction_forks_finds_only_the_compaction_child(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    # Not node 3 (assistant retry), not node 5 (duplicated tool result), not node 12
    # (compaction-shaped but in the subagent tree).
    assert compaction_forks(forest, children, tree) == [6]


def test_final_leaf_stays_in_the_main_tree(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    assert final_leaf(children, tree) == 9  # not 13, the subagent's higher-index leaf


def test_continue_seed_is_root_to_fork_child(forest):
    assert continue_seed(forest, 6) == [forest[0]["message"], forest[6]["message"]]


def test_recheck_seed_drops_truncated_assistant_and_appends_instruction(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    messages = recheck_seed(forest, children, tree, "Check your work.")
    # Final branch is [0, 6, 7, 8, 9]; node 9 (empty content, pending tool_calls) is dropped.
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[-1] == {"role": "user", "content": "Check your work."}
    assert messages[2]["tool_calls"]  # only the trailing assistant is stripped
    assert forest[9]["message"]["tool_calls"]  # the saved nodes are not mutated


def test_recheck_seed_strips_tool_calls_but_keeps_nonempty_assistant():
    nodes = [
        _node(None, False, {"role": "system", "content": "sys"}),
        _node(
            0,
            True,
            {
                "role": "assistant",
                "content": "final answer",
                "tool_calls": [{"id": "c1", "name": "search", "arguments": "{}"}],
            },
        ),
    ]
    children, _ = build_children(nodes)
    messages = recheck_seed(nodes, children, main_tree(children), "Check your work.")
    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert messages[1]["content"] == "final answer"
    assert messages[1]["tool_calls"] is None
    assert nodes[1]["message"]["tool_calls"]  # original untouched


def test_render_transcript_truncates_per_message(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    out = render_transcript(forest, children, tree, max_message_chars=50, max_total_chars=10_000)
    assert "[SYSTEM]" in out
    assert "[... truncated, 300 chars total]" in out  # node 8's 300-char tool result
    assert "x" * 51 not in out
    assert '[TOOL CALL] search({"q": "again"})' in out
    assert "elided" not in out


def test_render_transcript_elides_middle_under_total_budget(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    blocks = [_render_message(forest[i]["message"], 50) for i in (0, 6, 7, 8, 9)]
    # Enough for the head (system + task statement) and the last block, but not one more.
    budget = sum(len(b) + 2 for b in (blocks[0], blocks[1], blocks[4])) + 1
    out = render_transcript(forest, children, tree, max_message_chars=50, max_total_chars=budget)
    assert out == "\n\n".join([blocks[0], blocks[1], "[... 2 messages elided ...]", blocks[4]])


def test_usable_screens_bad_records(forest):
    assert usable({"nodes": forest, "errors": None})
    assert not usable({"nodes": forest, "errors": ["timeout"]})
    assert not usable({"nodes": [], "errors": None})
    unsampled = [_node(None, False, {"role": "system", "content": "sys"})]
    assert not usable({"nodes": unsampled, "errors": None})


# ------------------------------------------------------------------ verdict parsing


def test_parse_verdict():
    assert parse_verdict("Analysis...\nVERDICT: CORRECT") is True
    assert parse_verdict("VERDICT: INCORRECT") is False
    assert parse_verdict("verdict: correct") is True  # case-insensitive
    assert parse_verdict("VERDICT: CORRECT\n...wait, no.\nVERDICT: INCORRECT") is False  # last wins
    assert parse_verdict("the answer looks right to me") is None


def test_parse_verdict_skips_instruction_echo():
    echo = "answer with exactly `VERDICT: CORRECT` or `VERDICT: INCORRECT`."
    # A line quoting both options is an echo, not an answer: fall through to the real verdict...
    assert parse_verdict(f"VERDICT: CORRECT\n{echo}") is True
    # ...and an echo with no verdict anywhere else is unparseable.
    assert parse_verdict(echo) is None


# ------------------------------------------------------------------ config validation


def test_config_recheck_requires_inner():
    with pytest.raises(ValidationError, match="scored by the original taskset"):
        ReplayTasksetConfig(buffer_dir="/tmp/buf", mode="recheck")


def test_config_judge_forbids_inner():
    with pytest.raises(ValidationError, match="self-contained"):
        ReplayTasksetConfig(buffer_dir="/tmp/buf", mode="judge", inner={"id": "reverse-text-v1"})


def test_config_requires_buffer_dir():
    with pytest.raises(ValidationError, match="buffer_dir"):
        ReplayTasksetConfig(mode="judge")


def test_config_valid_judge_and_recheck():
    judge = ReplayTasksetConfig(buffer_dir="/tmp/buf", mode="judge")
    assert judge.inner is None
    recheck = ReplayTasksetConfig(buffer_dir="/tmp/buf", mode="recheck", inner={"id": "reverse-text-v1"})
    assert isinstance(recheck.inner, ReverseTextConfig)  # narrowed to the inner taskset's type


# ------------------------------------------------------------------ buffer


def _record(record_id: str, reward: float) -> dict:
    return {
        "id": record_id,
        "nodes": [
            _node(None, False, {"role": "system", "content": "sys"}),
            _node(0, False, {"role": "user", "content": f"task {record_id}"}),
            _node(1, True, {"role": "assistant", "content": f"answer {record_id}"}),
        ],
        "errors": None,
        "rewards": {"correct": reward},
        "stop_condition": "agent_completed",
        "task": {},
        "info": {},
    }


@pytest.fixture
def buffer_dir(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    """Two step dirs: step_1 is barrier-complete (train_rollouts.bin present), step_2 is
    not. step_2 also carries an errored record that must never become a candidate."""
    records = {
        "aaa": _record("aaa", 1.0),
        "bbb": _record("bbb", 1.0),
        "ccc": _record("ccc", 0.0),
        "ddd": _record("ddd", 1.0),
    }
    errored = _record("eee", 0.0) | {"errors": ["timeout"]}
    step_1 = tmp_path / "step_1"
    step_1.mkdir()
    (step_1 / "train_rollouts.jsonl").write_text("".join(json.dumps(records[i]) + "\n" for i in ("aaa", "bbb", "ccc")))
    (step_1 / "train_rollouts.bin").touch()
    step_2 = tmp_path / "step_2"
    step_2.mkdir()
    (step_2 / "train_rollouts.jsonl").write_text(json.dumps(records["ddd"]) + "\n" + json.dumps(errored) + "\n")
    return tmp_path, records


def _make_buffer(path: Path, mode: str, online: bool = False, **overrides) -> ReplayBuffer:
    kwargs = dict(
        buffer_dir=str(path),
        mode=mode,
        online=online,
        stop_conditions=None,
        source_envs=None,
        allow_container=False,
        success_threshold=0.5,
        balance_labels=True,
        max_candidates=4096,
        max_steps_back=None,
        seed=0,
    )
    kwargs.update(overrides)
    return ReplayBuffer(**kwargs)


def test_online_scan_requires_barrier(buffer_dir):
    path, _ = buffer_dir
    candidates = _make_buffer(path, "recheck", online=True).scan()
    assert {c.step for c in candidates} == {1}  # step_2 has no train_rollouts.bin yet


def test_offline_scan_skips_barrier_and_errored_records(buffer_dir):
    path, _ = buffer_dir
    candidates = _make_buffer(path, "recheck").scan()
    assert [c.source_id for c in candidates] == ["ddd", "aaa", "bbb", "ccc"]  # newest step first, no "eee"


def test_judge_balancing_interleaves_and_truncates(buffer_dir):
    path, _ = buffer_dir
    candidates = _make_buffer(path, "judge").scan()
    # 3 successes vs 1 failure: interleaved 1:1 and truncated to the smaller label.
    assert [c.original_reward for c in candidates] == [1.0, 0.0]


def test_pick_is_deterministic_and_wraps(buffer_dir):
    path, _ = buffer_dir
    buffer = _make_buffer(path, "recheck")
    buffer.scan()
    assert buffer.pick(2) == buffer.pick(2)
    assert buffer.pick(2) == buffer.pick(2 + len(buffer))


def test_read_record_round_trips(buffer_dir):
    path, records = buffer_dir
    buffer = _make_buffer(path, "recheck")
    for candidate in buffer.scan():
        assert asyncio.run(buffer.read_record(candidate)) == records[candidate.source_id]


def test_balance_is_a_view_not_attrition(buffer_dir):
    """Majority-label candidates dropped from one balanced view must pair up in a later
    one — rescans may not permanently destroy them."""
    path, _ = buffer_dir
    buffer = _make_buffer(path, "judge", online=True)
    assert [c.original_reward for c in buffer.scan()] == [1.0, 0.0]  # step_1 only: 2 pos, 1 neg
    step_3 = path / "step_3"
    step_3.mkdir()
    step_3.joinpath("train_rollouts.jsonl").write_text(
        "".join(json.dumps(_record(i, 0.0)) + "\n" for i in ("fff", "ggg"))
    )
    step_3.joinpath("train_rollouts.bin").touch()
    rebalanced = buffer.scan()
    # 2 pos + 3 neg total: both step_1 positives pair with negatives now.
    assert [c.original_reward for c in rebalanced] == [1.0, 0.0, 1.0, 0.0]


def test_replay_derived_records_skipped_by_default_but_listable(buffer_dir):
    """A "self" buffer sees the replay envs' own saved rollouts. By default replaying a
    replay is a feedback loop and is screened out; explicitly listing the replay env in
    source_envs is the deliberate opt-in for chained derivations."""
    path, _ = buffer_dir
    nested = _record("zzz", 1.0)
    nested["task"] = {"kind": "recheck", "source_task": {"prompt": "original"}, "prompt": None}
    nested["info"] = {"prime_rl": {"env_name": "replay-recheck"}}
    step_1 = path / "step_1"
    with open(step_1 / "train_rollouts.jsonl", "a") as f:
        f.write(json.dumps(nested) + "\n")
    by_default = _make_buffer(path, "judge", balance_labels=False).scan()
    assert "zzz" not in {c.source_id for c in by_default}
    opted_in = _make_buffer(path, "judge", balance_labels=False, source_envs=["replay-recheck"]).scan()
    assert {c.source_id for c in opted_in} == {"zzz"}


def test_unwrap_source_task_resolves_chains():
    original = {"prompt": "sort the list", "image": "sandbox:1"}
    depth_2 = {"kind": "recheck", "source_task": {"kind": "recheck", "source_task": original}}
    assert unwrap_source_task(depth_2) == original
    assert unwrap_source_task(original) == original  # non-derived tasks pass through
