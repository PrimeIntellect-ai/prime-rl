"""Trace to SFT rows: one row per trainable branch in the ``{messages, tools}`` form the
SFT loader reads, and ``export`` over a run's streams across resume attempts."""

import base64
import json

import orjson
import verifiers.v1 as vf
from verifiers.v1.graph import ParentLink

from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.sft import Images, export, sft_message, sft_tools, trace_rows
from prime_rl.utils.pathing import get_file_monitor_dir

PNG = base64.b64encode(b"\x89PNG fake").decode()
BASH = vf.Tool(name="bash", description="Run a command.", parameters={"type": "object", "properties": {}})


def _node(message, parent=None, **kw) -> vf.MessageNode:
    return vf.MessageNode(message=message, parent=parent, sampled=message.role == "assistant", **kw)


def _trace(reward: float = 1.0, *, compaction: str | None = None) -> vf.Trace:
    """system/user -> assistant tool call -> tool -> assistant; plus, when ``compaction``
    is given, a summary leaf off the first assistant turn linked as that kind."""
    nodes = [
        _node(vf.SystemMessage(content="be brief"), tools=[BASH]),
        _node(vf.UserMessage(content=[vf.TextContentPart(text="list "), vf.TextContentPart(text="files")]), 0),
        _node(
            vf.AssistantMessage(
                content=None,
                reasoning_content="look first",
                tool_calls=[vf.ToolCall(id="c1", name="bash", arguments='{"command": "ls"}')],
            ),
            1,
        ),
        _node(vf.ToolMessage(tool_call_id="c1", content="a.txt"), 2),
        _node(vf.AssistantMessage(content="a.txt"), 3),
    ]
    if compaction is not None:
        nodes.append(
            _node(
                vf.AssistantMessage(content="summary"),
                1,
                semantic_parents=[ParentLink(node=2, type=compaction)],
            )
        )
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="list files"), key="k0", hash="k0"),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="solver"),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
        info={"labels": {"stage": "solve"}},
        ok=True,
    )


def _episode(*traces: vf.Trace, ok: bool = True, env: str = "math", id: str | None = None) -> vf.WireEpisode:
    episode = vf.Episode(env=vf.EnvInfo(id=env, name=env), task=traces[0].task, traces=list(traces), ok=ok)
    if id is not None:
        episode.id = id
    return vf.WireEpisode.model_validate(episode.to_record())


def test_a_branch_becomes_one_row_in_the_sft_loader_form() -> None:
    episode = _episode(_trace(reward=0.5))
    [row] = trace_rows(episode, episode.traces[0])
    assert row["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "look first",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
        {"role": "assistant", "content": "a.txt"},
    ]
    assert row["tools"] == [
        {
            "type": "function",
            "function": {"name": "bash", "description": "Run a command.", "parameters": BASH.parameters},
        }
    ]
    assert row["meta"] == {
        "episode": episode.id,
        "trace": episode.traces[0].id,
        "branch": 0,
        "env": "math",
        "task": "k0",
        "agent": "solver",
        "reward": 0.5,
        "labels": {"stage": "solve"},
    }


def test_an_accepted_compaction_is_a_row_and_an_abandoned_attempt_is_not() -> None:
    accepted = _episode(_trace(compaction="compaction"))
    assert len(list(trace_rows(accepted, accepted.traces[0]))) == 2
    abandoned = _episode(_trace(compaction="compaction_attempt"))
    assert len(list(trace_rows(abandoned, abandoned.traces[0]))) == 1


def test_reasoning_details_become_reasoning_text() -> None:
    message = vf.AssistantMessage(content="ok", reasoning_content="why", provider_state=[{"type": "opaque"}])
    assert sft_message(message) == {"role": "assistant", "content": "ok", "reasoning_content": "why"}


def test_images_are_written_once_by_content(tmp_path) -> None:
    images = Images(tmp_path / "rows-images", tmp_path)
    url = f"data:image/png;base64,{PNG}"
    message = vf.UserMessage(
        content=[
            vf.TextContentPart(text="what is this"),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url)),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url)),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url="https://example.com/a.png")),
        ]
    )
    parts = sft_message(message, images)["content"]
    assert parts[1]["image_url"]["url"] == parts[2]["image_url"]["url"]
    assert (tmp_path / parts[1]["image_url"]["url"]).read_bytes() == base64.b64decode(PNG)
    assert parts[3]["image_url"]["url"] == "https://example.com/a.png"
    assert len(list((tmp_path / "rows-images").iterdir())) == 1
    # without Images the data URL stays inline
    assert sft_message(message)["content"][1]["image_url"]["url"] == url


def test_tools_leave_out_unset_strict() -> None:
    strict = BASH.model_copy(update={"strict": True})
    assert sft_tools([strict])[0]["function"]["strict"] is True
    assert "strict" not in sft_tools([BASH])[0]["function"]


def _land(run_dir, *episodes: vf.WireEpisode) -> None:
    stream = ChunkedJsonl(get_trace_stream(run_dir), max_bytes=1 << 20, compress=False)
    for episode in episodes:
        stream.append(orjson.dumps(episode.to_record(), option=orjson.OPT_APPEND_NEWLINE))
    stream.close()


def test_export_reads_every_attempt_and_filters(tmp_path) -> None:
    run = tmp_path / "run"
    _land(run, _episode(_trace(reward=1.0), id="retried", ok=False), _episode(_trace(reward=0.0), id="low"))
    current = get_file_monitor_dir(run)
    current.rename(current.with_name("file.attempt_1"))
    _land(run, _episode(_trace(reward=1.0), id="retried"), _episode(_trace(), id="other-env", env="code"))
    dest = tmp_path / "out" / "sft.jsonl"

    assert export([run], dest) == 3
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    # the retried episode's ok record (the last one) replaces its errored first attempt
    assert sorted(row["meta"]["episode"] for row in rows) == ["low", "other-env", "retried"]

    assert export([run], dest, envs=["math"], min_reward=0.5) == 1
    assert json.loads(dest.read_text())["meta"]["episode"] == "retried"
    assert export([run], dest, agents=["judge"]) == 0
    assert dest.read_text() == ""
    assert [path.name for path in dest.parent.iterdir()] == ["sft.jsonl"]
