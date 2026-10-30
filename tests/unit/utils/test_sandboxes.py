from prime_rl.utils import sandboxes


def test_clean_start_sweeps_the_previous_run_id(tmp_path, monkeypatch):
    swept = []

    async def sweep(labels, before, unique_labels):
        swept.extend(labels)

    monkeypatch.setattr(sandboxes, "sweep", sweep)
    monkeypatch.setattr(sandboxes, "_has_prime_key", lambda: True)

    sandboxes.sweep_previous_run(tmp_path)  # a run dir without an id has nothing to sweep
    run_id = sandboxes.stable_run_id(tmp_path)
    assert sandboxes.stable_run_id(tmp_path) == run_id
    sandboxes.sweep_previous_run(tmp_path)
    assert swept == [run_id]
