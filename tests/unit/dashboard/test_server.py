import orjson
import pytest

from prime_rl.dashboard import server


def write_records(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))


def configure_run(monkeypatch, tmp_path, records, *, step: int = 0):
    run_dir = tmp_path / "run"
    path = run_dir / "rollouts" / f"step_{step}" / "train" / "effective" / "traces.jsonl"
    write_records(path, records)
    monkeypatch.setattr(server, "output_dirs", [tmp_path])
    monkeypatch.setattr(server, "isolated", True)
    monkeypatch.setattr(server, "SIDECAR_DIR", tmp_path / "sidecars")
    server._run_registry.clear()
    return path


def record(episode_id: str, text: str = "evidence") -> dict:
    return {
        "id": episode_id,
        "env": {"id": "test"},
        "ok": True,
        "traces": [{"nodes": [{"message": {"role": "assistant", "content": text}, "sampled": True}]}],
    }


def test_larger_rewrite_invalidates_live_trace_caches(monkeypatch, tmp_path) -> None:
    path = configure_run(monkeypatch, tmp_path, [record("old-0"), record("old-1")])
    assert [row["id"] for row in server.episode_summaries(path)] == ["old-0", "old-1"]

    write_records(path, [record(f"new-{i}", "longer replacement evidence") for i in range(3)])

    assert [row["id"] for row in server.episode_summaries(path)] == ["new-0", "new-1", "new-2"]


def test_larger_rewrite_invalidates_summary_sidecar(monkeypatch, tmp_path) -> None:
    path = configure_run(monkeypatch, tmp_path, [record("old-0"), record("old-1")])
    server.episode_summaries(path)
    server._offsets_cache.pop(path, None)
    server._summaries_cache.pop(path, None)
    server._sidecar_written.pop(path, None)

    write_records(path, [record(f"new-{i}", "longer replacement evidence") for i in range(3)])

    assert [row["id"] for row in server.episode_summaries(path)] == ["new-0", "new-1", "new-2"]


def test_episode_filter_resolves_past_table_limit(monkeypatch, tmp_path) -> None:
    configure_run(monkeypatch, tmp_path, [record(f"episode-{i}") for i in range(5001)], step=9)

    result = server.list_episodes("run", 9, "train", "effective", limit=2, episode="episode-5000")

    assert result["total"] == 1
    assert result["episodes"][0]["line"] == 5000


def test_view_command_normalizes_and_validates_trace_indices(monkeypatch, tmp_path) -> None:
    configure_run(monkeypatch, tmp_path, [record("episode-0")])
    command = server.validate_view_command(
        {
            "run": "run",
            "tab": "traces",
            "step": "0",
            "kind": "train",
            "subset": "effective",
            "episode": "episode-0",
            "trace": 0,
            "highlight": [{"node": 0, "quote": "evidence"}],
        }
    )
    assert command["step"] == 0
    assert command["line"] == 0

    with pytest.raises(server.HTTPException, match="trace 1 not found") as error:
        server.validate_view_command(
            {
                "run": "run",
                "step": 0,
                "kind": "train",
                "subset": "effective",
                "line": 0,
                "trace": 1,
            }
        )
    assert error.value.status_code == 404


def test_view_command_rejects_malformed_numbers(monkeypatch, tmp_path) -> None:
    configure_run(monkeypatch, tmp_path, [record("episode-0")])

    with pytest.raises(server.HTTPException, match="step must be an integer") as error:
        server.validate_view_command({"run": "run", "step": "later", "kind": "train", "subset": "effective", "line": 0})
    assert error.value.status_code == 400
