"""Tests for PrimeMonitor — schema, serialization, outbox, and helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from prime_rl.utils.monitor.prime import (
    _SAMPLE_SCHEMA,
    _SAMPLE_SCHEMA_VERSION,
    _extract_tool_info,
    _flatten_completion_text,
    _serialize_rows_to_parquet,
    _write_atomic,
)

# ---------------------------------------------------------------------------
# _flatten_completion_text
# ---------------------------------------------------------------------------


def test_flatten_completion_text_none():
    assert _flatten_completion_text(None) == ""


def test_flatten_completion_text_empty_list():
    assert _flatten_completion_text([]) == ""


def test_flatten_completion_text_string():
    assert _flatten_completion_text("hello world") == "hello world"


def test_flatten_completion_text_chat_messages():
    messages = [
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "assistant", "content": "I'm sure of it."},
    ]
    result = _flatten_completion_text(messages)
    assert "The answer is 42." in result
    assert "I'm sure of it." in result


def test_flatten_completion_text_multipart_content():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is the image:"},
                {"type": "image_url", "image_url": {"url": "http://..."}},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]
    result = _flatten_completion_text(messages)
    assert "Here is the image:" in result
    assert "What do you see?" in result
    assert "image_url" not in result


def test_flatten_completion_text_mixed_types():
    messages = [
        {"role": "assistant", "content": "text message"},
        "plain string",
    ]
    result = _flatten_completion_text(messages)
    assert "text message" in result
    assert "plain string" in result


# ---------------------------------------------------------------------------
# _extract_tool_info
# ---------------------------------------------------------------------------


def test_extract_tool_info_no_tools():
    trajectory = [{"prompt": [], "completion": [{"role": "assistant", "content": "hello"}]}]
    count, names = _extract_tool_info(trajectory)
    assert count == 0
    assert names == []


def test_extract_tool_info_single_tool():
    trajectory = [
        {
            "prompt": [],
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "bash", "arguments": "ls"}],
                }
            ],
        }
    ]
    count, names = _extract_tool_info(trajectory)
    assert count == 1
    assert names == ["bash"]


def test_extract_tool_info_multiple_tools():
    trajectory = [
        {
            "prompt": [],
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "bash", "arguments": "ls"},
                        {"name": "python", "arguments": "print(1)"},
                    ],
                }
            ],
        },
        {
            "prompt": [],
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "bash", "arguments": "pwd"}],
                }
            ],
        },
    ]
    count, names = _extract_tool_info(trajectory)
    assert count == 3
    assert names == ["bash", "python"]  # sorted, deduplicated


def test_extract_tool_info_empty_tool_calls():
    trajectory = [{"prompt": [], "completion": [{"role": "assistant", "tool_calls": None}]}]
    count, names = _extract_tool_info(trajectory)
    assert count == 0
    assert names == []


def test_extract_tool_info_non_list_completion():
    trajectory = [{"prompt": [], "completion": "just a string"}]
    count, names = _extract_tool_info(trajectory)
    assert count == 0
    assert names == []


# ---------------------------------------------------------------------------
# _serialize_rows_to_parquet
# ---------------------------------------------------------------------------


def _make_sample_row(step: int = 0, idx: int = 0, reward: float = 0.5) -> dict:
    """Create a minimal valid row dict for testing."""
    return {
        "run_id": "test-run",
        "step": step,
        "schema_version": _SAMPLE_SCHEMA_VERSION,
        "tag": "",
        "problem_id": idx,
        "sample_id": idx,
        "prompt": json.dumps([{"role": "user", "content": "hello"}]),
        "completion": json.dumps([{"role": "assistant", "content": "world"}]),
        "completion_text": "world",
        "trajectory": json.dumps([]),
        "answer": "world",
        "task": "test-task",
        "info": "",
        "reward": reward,
        "advantage": None,
        "metrics": json.dumps({"accuracy": 1.0}),
        "timing": json.dumps({"generation_ms": 100}),
        "num_input_tokens": 5,
        "num_output_tokens": 3,
        "num_turns": 1,
        "num_tool_calls": 0,
        "tools_used": json.dumps([]),
        "is_completed": True,
        "is_truncated": False,
        "error": "",
        "created_at": datetime.now(timezone.utc),
    }


def test_serialize_rows_to_parquet_basic():
    rows = [_make_sample_row(step=0, idx=i) for i in range(5)]
    parquet_bytes = _serialize_rows_to_parquet(rows)

    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0


def test_serialize_rows_to_parquet_has_correct_schema():
    rows = [_make_sample_row()]
    parquet_bytes = _serialize_rows_to_parquet(rows)

    import io

    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.num_rows == 1
    assert set(_SAMPLE_SCHEMA.names).issubset(set(table.column_names))


def test_serialize_rows_to_parquet_preserves_data():
    rows = [_make_sample_row(step=42, idx=7, reward=0.99)]
    parquet_bytes = _serialize_rows_to_parquet(rows)

    import io

    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.column("step").to_pylist() == [42]
    assert table.column("sample_id").to_pylist() == [7]
    assert table.column("reward").to_pylist()[0] == pytest.approx(0.99)
    assert table.column("schema_version").to_pylist() == [_SAMPLE_SCHEMA_VERSION]


# ---------------------------------------------------------------------------
# _write_atomic
# ---------------------------------------------------------------------------


def test_write_atomic_creates_file(tmp_path: Path):
    target = tmp_path / "test.parquet"
    _write_atomic(target, b"hello world")
    assert target.exists()
    assert target.read_bytes() == b"hello world"


def test_write_atomic_no_tmp_left(tmp_path: Path):
    target = tmp_path / "test.parquet"
    _write_atomic(target, b"data")
    tmp_file = target.with_suffix(".parquet.tmp")
    assert not tmp_file.exists()


def test_write_atomic_overwrites_existing(tmp_path: Path):
    target = tmp_path / "test.parquet"
    target.write_bytes(b"old data")
    _write_atomic(target, b"new data")
    assert target.read_bytes() == b"new data"


# ---------------------------------------------------------------------------
# Filename pattern
# ---------------------------------------------------------------------------


def test_outbox_filename_pattern():
    from prime_rl.utils.monitor.prime import PrimeMonitor

    pattern = PrimeMonitor._OUTBOX_FILENAME_PATTERN
    assert pattern.match("step_0_chunk_0.parquet")
    assert pattern.match("step_100_chunk_3.parquet")
    assert pattern.match("step_9999_chunk_99.parquet")
    assert not pattern.match("step_0_abc123.parquet")
    assert not pattern.match("random_file.parquet")
    assert not pattern.match("step_0_chunk_0.parquet.tmp")
