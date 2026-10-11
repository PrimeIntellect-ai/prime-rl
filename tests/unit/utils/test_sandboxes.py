from types import SimpleNamespace

from prime_rl.utils import sandboxes


def test_clean_start_sweeps_the_previous_run_id(tmp_path, monkeypatch):
    deleted = []
    client = SimpleNamespace(
        bulk_delete=lambda labels: deleted.append(labels) or SimpleNamespace(succeeded=[], failed=[])
    )
    monkeypatch.setattr(sandboxes, "_client", lambda: client)

    sandboxes.sweep_previous_run(tmp_path)  # a run dir without an id has nothing to sweep
    run_id = sandboxes.stable_run_id(tmp_path)
    assert sandboxes.stable_run_id(tmp_path) == run_id
    sandboxes.sweep_previous_run(tmp_path)
    assert deleted == [[f"prl-run:{run_id}"]]
