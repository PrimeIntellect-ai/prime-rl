from concurrent.futures import ThreadPoolExecutor

import pytest

from prime_rl.utils.pathing import (
    ensure_checkpoint_generation,
    get_generation_path,
    read_checkpoint_generation,
    start_checkpoint_generation,
)


@pytest.mark.parametrize("hosted_id", [None, "hosted-run"])
def test_direct_components_share_persisted_identity(tmp_path, monkeypatch, hosted_id):
    monkeypatch.delenv("PRL_RUN_ID", raising=False)
    if hosted_id is not None:
        monkeypatch.setenv("PRL_RUN_ID", hosted_id)
    ckpt_dir = tmp_path / "checkpoints"
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(lambda _: ensure_checkpoint_generation(ckpt_dir), range(8)))
    assert all(state == states[0] for state in states)
    assert states[0]["generation"] not in ("", "legacy")
    assert states[0]["ceiling"] is None
    if hosted_id is not None:
        assert states[0]["generation"] == hosted_id
    assert read_checkpoint_generation(ckpt_dir) == states[0]


def test_component_startup_preserves_launcher_generation_and_resume_ceiling(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    start_checkpoint_generation(ckpt_dir, generation="saved-run", resume_step=40)
    before = get_generation_path(ckpt_dir).read_bytes()
    monkeypatch.setenv("PRL_RUN_ID", "different-runtime-id")
    assert ensure_checkpoint_generation(ckpt_dir) == {"generation": "saved-run", "ceiling": 40}
    assert get_generation_path(ckpt_dir).read_bytes() == before


def test_component_startup_does_not_repair_corrupt_identity(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = get_generation_path(ckpt_dir)
    path.write_text('{"generation":""}')
    with pytest.raises(ValueError, match="Invalid checkpoint generation"):
        ensure_checkpoint_generation(ckpt_dir)
    assert path.read_text() == '{"generation":""}'


def test_component_startup_does_not_upgrade_legacy_identity(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    start_checkpoint_generation(ckpt_dir, generation="legacy", resume_step=None)
    before = get_generation_path(ckpt_dir).read_bytes()
    monkeypatch.setenv("PRL_RUN_ID", "hosted-fresh-run")
    assert ensure_checkpoint_generation(ckpt_dir)["generation"] == "legacy"
    assert get_generation_path(ckpt_dir).read_bytes() == before
