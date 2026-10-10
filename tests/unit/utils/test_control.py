import os
import time

import msgspec
from fastapi.testclient import TestClient

from prime_rl.transports.batch import ControlTag, MicroBatch
from prime_rl.utils.control import ControlCommit, ControlPlane, create_app, paused_since, write_commit


def micro_batch(control: ControlTag | None = None) -> MicroBatch:
    return MicroBatch(
        input_ids=[1, 2],
        loss_mask=[True, True],
        advantages=[0.0, 0.0],
        inference_logprobs=[0.0, 0.0],
        position_ids=[0, 1],
        sequence_lengths=[2],
        temperatures=[1.0, 1.0],
        env_names=["env"],
        seq_lens=[2],
        control=control,
    )


def test_request_lifecycle(tmp_path):
    plane = ControlPlane(tmp_path)
    client = TestClient(create_app(plane))

    response = client.post("/v1/control", json={"action": "checkpoint"})
    assert response.status_code == 202
    record = response.json()
    assert (record["state"], record["step"]) == ("pending", None)
    assert client.post("/v1/control", json={"action": "pause"}).status_code == 409

    plane.accept(plane.take(), step=6)
    assert client.get(f"/v1/control/{record['id']}?wait=0.2").json()["state"] == "accepted"
    write_commit(tmp_path, ControlCommit(id=record["id"], action="checkpoint", step=6))
    assert record["id"] not in plane.reported
    assert client.get(f"/v1/control/{record['id']}").json() == {**record, "state": "committed", "step": 6}
    assert record["id"] in plane.reported


def test_pause_blocks_further_requests(tmp_path):
    plane = ControlPlane(tmp_path)
    client = TestClient(create_app(plane))
    plane.accept(plane.submit("pause"), step=16)
    assert client.post("/v1/control", json={"action": "checkpoint"}).status_code == 409


def test_invalid_and_unknown_requests(tmp_path):
    client = TestClient(create_app(ControlPlane(tmp_path)))
    assert client.post("/v1/control", json={"action": "explode"}).status_code == 422
    assert client.get("/v1/control/missing").status_code == 404
    assert client.get("/v1/control/missing?wait=600").status_code == 422


def test_paused_since_counts_only_new_pauses(tmp_path):
    write_commit(tmp_path, ControlCommit(id="old", action="pause", step=3))
    since = time.time()
    os.utime(tmp_path / "old.json", (since - 10, since - 10))
    write_commit(tmp_path, ControlCommit(id="saved", action="checkpoint", step=9))
    assert not paused_since(tmp_path, since)
    write_commit(tmp_path, ControlCommit(id="new", action="pause", step=9))
    assert paused_since(tmp_path, since)


def test_control_tag_survives_the_wire():
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
    tag = ControlTag(id="x", action="pause")
    [tagged, untagged] = decoder.decode(msgspec.msgpack.encode([micro_batch(tag), micro_batch()]))
    assert tagged.control == tag
    assert untagged.control is None
