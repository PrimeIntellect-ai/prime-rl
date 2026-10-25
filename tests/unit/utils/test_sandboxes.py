from prime_rl.utils import sandboxes


def test_clean_start_sweeps_the_previous_run_id(tmp_path, monkeypatch):
    swept = []

    async def sweep_run(run_id, before):
        swept.append(run_id)

    monkeypatch.setattr(sandboxes, "sweep_run", sweep_run)

    sandboxes.sweep_previous_run(tmp_path)  # a run dir without an id has nothing to sweep
    run_id = sandboxes.stable_run_id(tmp_path)
    assert sandboxes.stable_run_id(tmp_path) == run_id
    sandboxes.sweep_previous_run(tmp_path)
    assert swept == [run_id]
