from uuid import uuid4

import msgspec
import pytest

from prime_rl.transports.batch.journal import BatchCheckpoint, BatchJournal


def test_round_trip_and_corruption(tmp_path):
    journal = BatchJournal(tmp_path)
    batch_id = f"{uuid4().hex}:1"
    payloads = [b"first rank", b"second rank"]
    journal.write(batch_id, payloads)
    assert BatchJournal(tmp_path).read(batch_id) == payloads
    with pytest.raises(OSError):
        journal.write(batch_id, [b"replacement"])
    assert journal.read(batch_id) == payloads
    (journal.batch_path(batch_id) / "rank_1.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        journal.read(batch_id)


def test_commit_releases_only_covered_batches_in_its_session(tmp_path):
    journal = BatchJournal(tmp_path)
    session = uuid4().hex
    ids = [f"{session}:{step}" for step in (1, 2, 3)]
    other = f"{uuid4().hex}:1"
    for batch_id in [*ids, other]:
        journal.write(batch_id, [batch_id.encode()])
    checkpoint = tmp_path / "checkpoints" / "step_2" / "trainer"
    checkpoint.mkdir(parents=True)
    journal.commit_checkpoint(checkpoint, 2, ids[1])
    assert not journal.batch_path(ids[0]).exists()
    assert not journal.batch_path(ids[1]).exists()
    assert journal.read(ids[2]) == [ids[2].encode()]
    assert journal.read(other) == [other.encode()]
    committed = msgspec.json.decode((checkpoint / "batch.json").read_bytes(), type=BatchCheckpoint)
    assert committed.batch.batch_id == ids[1]
    # Reconcile a lost notification / interrupted cleanup without rewriting the checkpoint.
    BatchJournal(tmp_path).prune(checkpoint)


def test_failed_commit_does_not_release_batches(tmp_path, monkeypatch):
    journal = BatchJournal(tmp_path)
    batch_id = f"{uuid4().hex}:1"
    journal.write(batch_id, [b"retained"])
    checkpoint = tmp_path / "checkpoints" / "step_1" / "trainer"
    checkpoint.mkdir(parents=True)

    def fail_write(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("prime_rl.transports.batch.journal._write_atomic", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        journal.commit_checkpoint(checkpoint, 1, batch_id)
    assert not (checkpoint / "batch.json").exists()
    assert journal.read(batch_id) == [b"retained"]


def test_interrupted_payload_write_is_not_published(tmp_path, monkeypatch):
    journal = BatchJournal(tmp_path)
    batch_id = f"{uuid4().hex}:1"

    def fail_write(path, payload):
        path.write_bytes(payload[:1])
        raise OSError("interrupted write")

    monkeypatch.setattr("prime_rl.transports.batch.journal._write_file", fail_write)
    with pytest.raises(OSError, match="interrupted write"):
        journal.write(batch_id, [b"incomplete"])
    assert not journal.batch_path(batch_id).exists()
    with pytest.raises(FileNotFoundError):
        journal.read(batch_id)


def test_failed_deletion_can_be_reconciled(tmp_path, monkeypatch):
    journal = BatchJournal(tmp_path)
    batch_id = f"{uuid4().hex}:1"
    journal.write(batch_id, [b"retained"])
    checkpoint = tmp_path / "checkpoints" / "step_1" / "trainer"
    checkpoint.mkdir(parents=True)

    with monkeypatch.context() as patch:

        def fail_delete(path):
            raise OSError("interrupted deletion")

        patch.setattr("prime_rl.transports.batch.journal.shutil.rmtree", fail_delete)
        with pytest.raises(OSError, match="interrupted deletion"):
            journal.commit_checkpoint(checkpoint, 1, batch_id)
    assert (checkpoint / "batch.json").exists()
    assert journal.read(batch_id) == [b"retained"]
    BatchJournal(tmp_path).prune(checkpoint)
    assert not journal.batch_path(batch_id).exists()
