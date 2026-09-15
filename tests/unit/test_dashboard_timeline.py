from prime_rl.dashboard.server import project_episode_timeline


def _node(
    parent: int | None,
    timestamp: float,
    semantic_parents: list[dict] | None = None,
    message: dict | None = None,
) -> dict:
    return {
        "parent": parent,
        "semantic_parents": semantic_parents or [],
        "timestamp": timestamp,
        "message": message or {"role": "assistant", "content": f"node {timestamp}"},
    }


def _call(
    node: int,
    start: float,
    end: float,
    *,
    prompt_tokens: int = 100,
    cached_tokens: int = 20,
) -> dict:
    return {
        "node": node,
        "time": {"start": start, "end": end},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "cached_input_tokens": cached_tokens,
            "completion_tokens": 10,
        },
    }


def _episode(nodes: list[dict], calls: list[dict]) -> dict:
    return {
        "traces": [
            {
                "nodes": nodes,
                "calls": calls,
                "is_completed": True,
                "agent": {"name": "root", "config": {"model": "test-model"}},
                "timing": {"agent": {"start": 0.0, "end": 10.0}},
            }
        ]
    }


def test_semantic_timeline_projects_concurrent_fork_join() -> None:
    nodes = [
        _node(None, 0.0),
        _node(0, 2.0),
        _node(0, 4.0, [{"node": 1, "type": "subagent_call"}]),
        _node(0, 4.1, [{"node": 1, "type": "subagent_call"}]),
        _node(
            1,
            7.0,
            [
                {"node": 1, "type": "continuation"},
                {"node": 2, "type": "subagent_return"},
                {"node": 3, "type": "subagent_return"},
                {"node": 2, "type": "vendor:review"},
            ],
        ),
    ]
    calls = [
        _call(1, 1.0, 2.0),
        _call(2, 3.0, 4.0),
        _call(3, 3.1, 4.1),
        _call(4, 6.0, 7.0),
    ]

    timeline = project_episode_timeline(_episode(nodes, calls))

    assert [lane["label"] for lane in timeline["semantic_lanes"]] == [
        "root",
        "root · context 0",
        "subagent 1 · context 0",
        "subagent 2 · context 0",
    ]
    assert [lane["usage"]["model_calls"] for lane in timeline["semantic_lanes"][1:]] == [
        2,
        1,
        1,
    ]
    assert [lane["usage"]["total_tokens"] for lane in timeline["semantic_lanes"][1:]] == [
        260,
        130,
        130,
    ]
    assert [lane["usage"]["latest_context_tokens"] for lane in timeline["semantic_lanes"][1:]] == [
        120,
        120,
        120,
    ]
    assert [lane["context"] for lane in timeline["semantic_lanes"][1:]] == [
        {"agent": "root", "index": 0},
        {"agent": "subagent 1", "index": 0},
        {"agent": "subagent 2", "index": 0},
    ]
    assert [edge["type"] for edge in timeline["semantic_edges"]] == [
        "subagent_call",
        "subagent_call",
        "continuation",
        "subagent_return",
        "subagent_return",
        "vendor:review",
    ]


def test_semantic_timeline_starts_new_context_after_compaction() -> None:
    nodes = [
        _node(None, 0.0),
        _node(0, 2.0),
        _node(1, 4.0, [{"node": 1, "type": "continuation"}]),
        _node(2, 6.0, [{"node": 2, "type": "compaction"}]),
    ]
    calls = [
        _call(1, 1.0, 2.0, prompt_tokens=150),
        _call(2, 3.0, 4.0),
        _call(3, 5.0, 6.0, prompt_tokens=50, cached_tokens=10),
    ]

    timeline = project_episode_timeline(_episode(nodes, calls))

    assert [lane["label"] for lane in timeline["semantic_lanes"]] == [
        "root",
        "root · context 0",
        "root · context 1",
    ]
    assert [lane["usage"]["model_calls"] for lane in timeline["semantic_lanes"][1:]] == [
        2,
        1,
    ]
    assert [lane["usage"]["total_tokens"] for lane in timeline["semantic_lanes"][1:]] == [
        310,
        70,
    ]
    assert [lane["usage"]["latest_context_tokens"] for lane in timeline["semantic_lanes"][1:]] == [
        120,
        60,
    ]
    assert [lane["usage"]["max_context_tokens"] for lane in timeline["semantic_lanes"][1:]] == [
        170,
        60,
    ]
    assert [lane["context"] for lane in timeline["semantic_lanes"][1:]] == [
        {"agent": "root", "index": 0},
        {"agent": "root", "index": 1},
    ]
    assert [edge["type"] for edge in timeline["semantic_edges"]] == [
        "continuation",
        "compaction",
    ]


def test_semantic_timeline_keeps_rejected_and_accepted_compaction_attempts() -> None:
    nodes = [
        _node(None, 0.0),
        _node(0, 2.0),
        _node(1, 4.0, [{"node": 1, "type": "compaction_attempt"}]),
        _node(1, 5.0, [{"node": 1, "type": "compaction_attempt"}]),
        _node(0, 7.0, [{"node": 3, "type": "compaction"}]),
    ]
    nodes[2]["mask"] = [True]
    nodes[3]["mask"] = [True]
    calls = [
        _call(1, 1.0, 2.0),
        _call(2, 3.0, 4.0),
        _call(3, 4.0, 5.0),
        _call(4, 6.0, 7.0),
    ]

    timeline = project_episode_timeline(_episode(nodes, calls))
    attempts = [lane for lane in timeline["semantic_lanes"] if lane.get("compaction_attempt")]

    assert [attempt["compaction_attempt"]["accepted"] for attempt in attempts] == [
        False,
        True,
    ]
    assert [attempt["spans"][1]["trainable"] for attempt in attempts] == [
        True,
        True,
    ]
    assert [attempt["context"] for attempt in attempts] == [
        {"agent": "root", "index": 0},
        {"agent": "root", "index": 0},
    ]
    assert timeline["semantic_lanes"][-1]["context"] == {
        "agent": "root",
        "index": 1,
    }


def test_semantic_timeline_falls_back_to_physical_only() -> None:
    nodes = [_node(None, 0.0), _node(0, 2.0)]
    timeline = project_episode_timeline(_episode(nodes, [_call(1, 1.0, 2.0)]))

    assert timeline["lanes"]
    assert timeline["semantic_lanes"] == []
    assert timeline["semantic_edges"] == []


def test_semantic_timeline_recovers_interrupted_tool_continuation() -> None:
    nodes = [
        _node(None, 0.0),
        _node(0, 2.0),
        _node(
            0,
            4.0,
            [{"node": 1, "type": "subagent_call"}],
            {"role": "assistant", "tool_calls": [{"id": "first-tool"}]},
        ),
        _node(
            2,
            5.0,
            message={"role": "tool", "tool_call_id": "first-tool", "content": "result"},
        ),
        _node(
            3,
            6.0,
            [{"node": 2, "type": "continuation"}],
            {"role": "assistant", "tool_calls": [{"id": "interrupted-tool"}]},
        ),
        _node(
            4,
            7.0,
            message={"role": "tool", "tool_call_id": "interrupted-tool", "content": "result"},
        ),
        _node(5, 8.0),
    ]
    calls = [
        _call(1, 1.0, 2.0),
        _call(2, 3.0, 4.0),
        _call(4, 5.0, 6.0),
        _call(6, 7.0, 8.0),
    ]

    timeline = project_episode_timeline(_episode(nodes, calls))

    assert [lane["label"] for lane in timeline["semantic_lanes"]] == [
        "root",
        "root · context 0",
        "subagent 1 · context 0",
    ]
    assert timeline["semantic_lanes"][2]["usage"]["model_calls"] == 3
    assert timeline["semantic_edges"][-1] == {
        "trace_index": 0,
        "source_node": 4,
        "target_node": 6,
        "type": "continuation",
        "inferred": True,
    }


def test_semantic_timeline_labels_unrecoverable_components_as_unlinked() -> None:
    nodes = [
        _node(None, 0.0),
        _node(0, 2.0),
        _node(1, 4.0, [{"node": 1, "type": "continuation"}]),
        _node(0, 6.0),
    ]
    timeline = project_episode_timeline(_episode(nodes, [_call(1, 1.0, 2.0), _call(2, 3.0, 4.0), _call(3, 5.0, 6.0)]))

    assert [lane["label"] for lane in timeline["semantic_lanes"]] == [
        "root",
        "root · context 0",
        "unlinked 1 · context 0",
    ]
    assert timeline["semantic_lanes"][2]["context"] == {
        "agent": "unlinked 1",
        "index": 0,
        "unlinked": True,
    }
