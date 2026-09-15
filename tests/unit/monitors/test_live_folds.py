"""The live-trace reader folds partial files and survives the ways a file can be broken."""

import orjson

from prime_rl.monitors.file.traces.live import LiveFolds, read_live


def delta(trace: str, **fields) -> bytes:
    return orjson.dumps({"trace": trace, **fields}) + b"\n"


def header(trace: str) -> bytes:
    return delta(
        trace,
        open={"id": trace, "version": 1, "verifiers": "x", "task": {}},
        dispatch={"id": "d1", "kind": "eval", "env": "e", "started": 1.0},
    )


def test_incremental_fold_reads_only_the_tail_and_waits_for_torn_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(header("t") + delta("t", nodes=[{"message": {"role": "user", "content": "q"}}]))
    folds = LiveFolds()
    dispatch, trace = folds.read(path)
    assert dispatch["kind"] == "eval" and len(trace["nodes"]) == 1
    # a torn append: the whole line waits, the rest is applied
    torn = delta("t", nodes=[{"message": {"role": "assistant", "content": "a"}}])
    with path.open("ab") as f:
        f.write(torn[:-5])
    _, trace = folds.read(path)
    assert len(trace["nodes"]) == 1
    with path.open("ab") as f:
        f.write(torn[-5:])
    _, trace = folds.read(path)
    assert len(trace["nodes"]) == 2
    # the file vanishing (its episode landed) drops the fold
    path.unlink()
    assert folds.read(path) is None


def test_headerless_file_folds_to_nothing(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(delta("t", nodes=[{"message": {"role": "user", "content": "q"}}]))
    assert read_live(path) is None
    assert LiveFolds().read(path) is None
