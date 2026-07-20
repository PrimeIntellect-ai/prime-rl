"""Shape-sniffing and selection in ``scripts/export_sft.py``: a traces.jsonl line may be
an episode (verifiers v1 eval runs) or a flat trace record (prime-rl's rollout dumps),
a malformed line reports its file:line, and untrainable traces drop unless
``--include-untrainable``."""

import importlib.util
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_sft.py"
_spec = importlib.util.spec_from_file_location("export_sft", _SCRIPT)
assert _spec is not None and _spec.loader is not None
export_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_sft)


def _trace(*, trainable: bool = True, content: str = "hi") -> vf.WireTrace:
    return vf.WireTrace(
        task=vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0)),
        nodes=[MessageNode(message=AssistantMessage(content=content), sampled=True)],
        trainable=trainable,
    )


def _episode_line(trace: vf.WireTrace) -> str:
    return vf.WireEpisode(env="dummy", task=trace.task, traces=[trace]).model_dump_json()


def test_iter_traces_sniffs_episode_and_flat_lines(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        _episode_line(_trace(content="from-episode")) + "\n\n" + _trace(content="flat").model_dump_json() + "\n"
    )
    traces = list(export_sft.iter_traces(path))
    assert [t.last_reply for t in traces] == ["from-episode", "flat"]


def test_iter_traces_reports_file_and_line_on_malformed_lines(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(_trace().model_dump_json() + "\n{not json\n")
    with pytest.raises(SystemExit, match=r"traces\.jsonl:2"):
        list(export_sft.iter_traces(path))

    path.write_text('{"neither": "episode nor trace"}\n')
    with pytest.raises(SystemExit, match=r"traces\.jsonl:1"):
        list(export_sft.iter_traces(path))


def test_untrainable_traces_drop_by_default(tmp_path, capsys):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        _trace(content="policy").model_dump_json()
        + "\n"
        + _trace(trainable=False, content="judge").model_dump_json()
        + "\n"
    )

    rows = export_sft.collect_rows(path, min_reward=None, drop_truncated=False, include_untrainable=False)
    assert [row["messages"][0]["content"] for row in rows] == ["policy"]
    assert "dropped 1 untrainable trace(s)" in capsys.readouterr().out

    rows = export_sft.collect_rows(path, min_reward=None, drop_truncated=False, include_untrainable=True)
    assert [row["messages"][0]["content"] for row in rows] == ["policy", "judge"]
