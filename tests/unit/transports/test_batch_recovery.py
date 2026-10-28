from uuid import uuid4

import msgspec
import pytest

from prime_rl.transports.batch.journal import BatchCheckpoint, BatchJournal, BatchRecord
from prime_rl.transports.batch.recovery import BatchRequest, TrainerGroup, TrainerHandshake, TrainerHello, plan_replay


def test_handshake_counts_distinct_ranks_and_fences_connections():
    state = TrainerHandshake(data_world_size=2)
    group = TrainerGroup(uuid4().hex, 4, 1)
    first = TrainerHello(group, 0, 0)
    assert not state.add(b"rank0", first)
    assert not state.add(b"rank0", first)
    with pytest.raises(ValueError, match="incomplete"):
        state.accept()
    with pytest.raises(ValueError, match="Duplicate"):
        state.add(b"other", first)
    with pytest.raises(ValueError, match="another trainer rank"):
        state.add(b"rank0", TrainerHello(group, 1, 0))
    for rank in range(1, 4):
        assert state.add(f"rank{rank}".encode(), TrainerHello(group, rank, rank // 2)) == (rank == 3)
    state.accept()
    assert state.validate_request(b"rank3", BatchRequest(group.attempt_id, 3, 1), 1) == 1
    with pytest.raises(ValueError, match="connection"):
        state.validate_request(b"other", BatchRequest(group.attempt_id, 3, 1), 1)
    with pytest.raises(ValueError, match="step"):
        state.validate_request(b"rank3", BatchRequest(group.attempt_id, 3, 0), 1)

    replacement = TrainerGroup(uuid4().hex, 4, 1)
    for rank in range(4):
        state.add(f"new{rank}".encode(), TrainerHello(replacement, rank, rank // 2))
    # A candidate does not replace the active attempt until explicitly accepted.
    assert state.active == group
    state.accept()
    with pytest.raises(ValueError, match="accepted trainer attempt"):
        state.validate_request(b"rank0", BatchRequest(group.attempt_id, 0, 1), 1)
    with pytest.raises(ValueError, match="retired"):
        state.add(b"rank0", first)


@pytest.mark.parametrize("change", ["attempt", "world_size", "start_step", "restored", "data_rank", "rank"])
def test_handshake_rejects_inconsistent_ranks(change):
    checkpoint = BatchCheckpoint(0, BatchRecord(f"{uuid4().hex}:0", ["a" * 64, "b" * 64]))
    group = TrainerGroup(uuid4().hex, 4, 1, checkpoint)
    state = TrainerHandshake(2)
    state.add(b"0", TrainerHello(group, 0, 0))
    changes = {
        "attempt": {"attempt_id": uuid4().hex},
        "world_size": {"world_size": 6},
        "start_step": {"start_step": 2},
        "restored": {"restored": BatchCheckpoint(0, BatchRecord(f"{uuid4().hex}:0", ["a" * 64, "b" * 64]))},
    }
    other = msgspec.structs.replace(group, **changes.get(change, {}))
    hello = TrainerHello(other, 5 if change == "rank" else 1, 1 if change == "data_rank" else 0)
    with pytest.raises(ValueError):
        state.add(b"1", hello)
    assert state.active is None and len(state.pending) == 1


@pytest.fixture
def retained_batches(tmp_path):
    journal = BatchJournal(tmp_path)
    session = uuid4().hex
    records = [journal.write(f"{session}:{step}", [f"{step}:0".encode(), f"{step}:1".encode()]) for step in range(4)]
    path = tmp_path / "checkpoints" / "step_1" / "trainer"
    path.mkdir(parents=True)
    journal.commit_checkpoint(path, 1, records[1].batch_id)
    checkpoint = msgspec.json.decode((path / "batch.json").read_bytes(), type=BatchCheckpoint)
    return journal, session, TrainerGroup(uuid4().hex, 4, 2, checkpoint), records


def test_replay_plan_starts_after_checkpoint_and_includes_pending_batch(retained_batches):
    journal, session, group, records = retained_batches
    other_session = uuid4().hex
    journal.write(f"{other_session}:2", [b"unrelated"])
    plan = plan_replay(journal, group, session, through=3, data_world_size=2)
    assert plan.restored == group.restored and plan.batches == records[2:]
    assert not journal.batch_path(records[1].batch_id).exists()
    assert journal.read(records[2].batch_id) == [b"2:0", b"2:1"]


@pytest.mark.parametrize("fault", ["missing", "corrupt", "foreign", "topology", "checkpoint", "no_checkpoint"])
def test_replay_plan_fails_closed(retained_batches, fault):
    journal, session, group, records = retained_batches
    if fault == "missing":
        (journal.batch_path(records[2].batch_id) / "manifest.json").unlink()
    elif fault == "corrupt":
        (journal.batch_path(records[2].batch_id) / "rank_1.bin").write_bytes(b"changed")
    elif fault == "foreign":
        session = uuid4().hex
    elif fault == "topology":
        record = BatchRecord(records[2].batch_id, records[2].hashes[:1])
        (journal.batch_path(record.batch_id) / "manifest.json").write_bytes(msgspec.json.encode(record))
    elif fault == "checkpoint":
        changed = BatchCheckpoint(1, BatchRecord(records[1].batch_id, ["0" * 64, "1" * 64]))
        group = msgspec.structs.replace(group, restored=changed)
    elif fault == "no_checkpoint":
        group = msgspec.structs.replace(group, restored=None)
    with pytest.raises((ValueError, FileNotFoundError)):
        plan_replay(journal, group, session, through=3, data_world_size=2)
