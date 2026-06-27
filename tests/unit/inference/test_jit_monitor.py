import json

from prime_rl.inference.vllm import jit_monitor


class _FakeKernel:
    name = "_fake_kernel"


class _LongRepr:
    def __repr__(self):
        return "x" * (jit_monitor._MAX_DETAIL_REPR_CHARS + 10)


def test_compile_details_preserves_scalars_and_truncates_repr():
    details = jit_monitor._compile_details(
        {
            "fn": _FakeKernel(),
            "key": ("shape", 1, 20),
            "already_compiled": False,
            "is_manual_warmup": True,
            "repr": _LongRepr(),
            "ignored": "value",
        }
    )

    assert details["key"] == "('shape', 1, 20)"
    assert details["already_compiled"] is False
    assert details["is_manual_warmup"] is True
    assert details["repr"].endswith("...")
    assert len(details["repr"]) == jit_monitor._MAX_DETAIL_REPR_CHARS
    assert "ignored" not in details


def test_append_jit_event_writes_details(tmp_path, monkeypatch):
    path = tmp_path / "jit.jsonl"
    monkeypatch.setenv("PRIME_RL_JIT_MONITOR_LOG", str(path))
    jit_monitor._log_path.cache_clear()

    jit_monitor._append_jit_event(
        {
            "fn": _FakeKernel(),
            "key": ("shape", 1, 20),
            "already_compiled": False,
        }
    )

    record = json.loads(path.read_text().strip())
    assert record["kernel"] == "_fake_kernel"
    assert record["kwargs"] == ["already_compiled", "fn", "key"]
    assert record["details"] == {
        "already_compiled": False,
        "key": "('shape', 1, 20)",
    }
