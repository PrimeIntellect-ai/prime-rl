import orjson

from prime_rl.dashboard.server import project_episode_timeline


def tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "name": "ipython",
        "arguments": orjson.dumps({"code": code}).decode(),
    }


def node(role: str, timestamp: float, parent: int | None = None, **message) -> dict:
    return {
        "parent": parent,
        "timestamp": timestamp,
        "message": {"role": role, **message},
    }


def test_timeline_infers_recursive_rlm_hierarchy_with_concurrent_siblings():
    nodes = [
        node("system", 0, content="Conversation log: /tmp/run/agent/sessions/root.jsonl"),
        node("user", 0, 0, content="root"),
        node(
            "assistant",
            2,
            1,
            content="",
            tool_calls=[tool_call("main-rlm", "await gather(*[rlm(task) for task in tasks])")],
        ),
        node(
            "system",
            3,
            content="Conversation log: /tmp/run/agent/session-artifacts/root/sub-a/child-a.jsonl",
        ),
        node("user", 3, 3, content="child A"),
        node(
            "assistant",
            5,
            4,
            content="",
            tool_calls=[tool_call("nested-rlm", "await rlm(next_task)")],
        ),
        node(
            "system",
            6,
            content=(
                "Conversation log: /tmp/run/agent/session-artifacts/root/sub-a/"
                "session-artifacts/child-a/sub-g/grandchild.jsonl"
            ),
        ),
        node("user", 6, 6, content="grandchild"),
        node("assistant", 8, 7, content="done"),
        node("tool", 9, 5, content="grandchild result", tool_call_id="nested-rlm"),
        node(
            "system",
            3.5,
            content="Conversation log: /tmp/run/agent/session-artifacts/root/sub-b/child-b.jsonl",
        ),
        node("user", 3.5, 10, content="child B"),
        node("assistant", 7, 11, content="done"),
        node("tool", 12, 2, content="children result", tool_call_id="main-rlm"),
    ]
    trace = {
        "nodes": nodes,
        "calls": [
            {"node": 2, "time": {"start": 1, "end": 2}},
            {"node": 5, "time": {"start": 4, "end": 5}},
            {"node": 8, "time": {"start": 7, "end": 8}},
            {"node": 12, "time": {"start": 6, "end": 7}},
        ],
        "is_completed": True,
        "ok": True,
        "agent": {"name": "rlm"},
        "timing": {"start": 0},
    }

    concurrent_trace = {
        "nodes": [
            node("user", 5.5, content="review concurrently"),
            node("assistant", 5.8, 0, content="reviewed"),
        ],
        "calls": [{"node": 1, "time": {"start": 5.6, "end": 5.8}}],
        "is_completed": True,
        "ok": True,
        "agent": {"name": "reviewer"},
        "timing": {"start": 5.5},
    }

    lanes = project_episode_timeline({"traces": [trace, concurrent_trace]})["lanes"]

    assert [lane["id"] for lane in lanes] == [
        "trace-0",
        "trace-0-subagent-3",
        "trace-0-subagent-6",
        "trace-0-subagent-10",
        "trace-1",
    ]
    lanes_by_id = {lane["id"]: lane for lane in lanes}
    assert lanes_by_id["trace-0-subagent-3"]["parent_id"] == "trace-0"
    assert lanes_by_id["trace-0-subagent-3"]["depth"] == 1
    assert lanes_by_id["trace-0-subagent-6"]["parent_id"] == "trace-0-subagent-3"
    assert lanes_by_id["trace-0-subagent-6"]["depth"] == 2
    assert lanes_by_id["trace-0-subagent-10"]["parent_id"] == "trace-0"
    assert lanes_by_id["trace-0-subagent-10"]["depth"] == 1
