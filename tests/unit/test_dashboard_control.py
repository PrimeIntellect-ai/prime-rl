"""The dashboard's run control proxy (``--allow-control``) against a fake control
endpoint, and trace_id-only report citations resolved through the episodes lookup."""

import json
import os
import socket
import threading

import pytest
import zmq
from fastapi.testclient import TestClient

from prime_rl.dashboard import server
from prime_rl.eval.control import CONTROL_FILE

STATUS = {
    "episodes": [
        {
            "task_key": "job-1/solve/0",
            "env": "prime_flow",
            "group": "g",
            "seats": ["solver"],
            "traces": [{"trace_id": "tr-1", "seat": "solver", "stage": "running", "turns": 3}],
        }
    ],
    "inflight": 1,
    "max_inflight": 4,
    "min_inflight": 1,
}


class FakeControl:
    """A REP socket speaking the run control's JSON frames, recording each request."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.address = f"tcp://127.0.0.1:{port}"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def reply(self, request: dict) -> dict:
        method, args = request["method"], request["args"]
        if method == "status":
            return {"ok": True, **STATUS}
        if method == "steer":
            if args.get("trace_id") == "gone":
                return {"ok": False, "error": "ValueError: no such trace"}
            return {
                "ok": True,
                "receipts": [
                    {
                        "queued": True,
                        "reason": None,
                        "delivery": "native",
                        "trace_id": args.get("trace_id"),
                        "seat": "solver",
                    }
                ],
            }
        return {"ok": False, "error": f"unknown request {method!r}"}

    def serve(self) -> None:
        while not self.stop.is_set():
            try:
                request = json.loads(self.socket.recv())
            except zmq.Again:
                continue
            self.requests.append(request)
            self.socket.send(json.dumps(self.reply(request)).encode())

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        self.socket.close()
        self.ctx.term()


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "output_dirs", [tmp_path])
    monkeypatch.setattr(server, "isolated", True)
    monkeypatch.setattr(server, "_run_registry", {})
    monkeypatch.setattr(server, "allow_control", True)
    return tmp_path


@pytest.fixture
def control():
    fake = FakeControl()
    yield fake
    fake.close()


def make_run(outputs, name: str = "eval-run", control_info: dict | None = None):
    run_dir = outputs / name
    (run_dir / "configs").mkdir(parents=True)
    if control_info is not None:
        (run_dir / CONTROL_FILE).write_text(json.dumps(control_info))
    return run_dir


def here(address: str) -> dict:
    return {"address": address, "host": socket.gethostname(), "pid": os.getpid()}


def test_control_is_off_without_the_flag(outputs, control, monkeypatch):
    monkeypatch.setattr(server, "allow_control", False)
    make_run(outputs, control_info=here(control.address))
    client = TestClient(server.app)
    assert client.get("/api/control").json() == {"enabled": False}
    assert client.get("/api/runs/eval-run/control").status_code == 403
    response = client.post("/api/runs/eval-run/control/steer", json={"text": "hi", "trace_id": "tr-1"})
    assert response.status_code == 403
    assert control.requests == []


def test_control_needs_a_live_run_on_this_host(outputs, control):
    make_run(outputs, "no-control")
    make_run(outputs, "elsewhere", {"address": control.address, "host": "not-" + socket.gethostname(), "pid": 1})
    make_run(outputs, "dead", {"address": control.address, "host": socket.gethostname(), "pid": 2**22 + 1})
    client = TestClient(server.app)
    assert client.get("/api/control").json() == {"enabled": True}
    for run, reason in (("no-control", "is the eval running"), ("elsewhere", "not on this host"), ("dead", "gone")):
        response = client.get(f"/api/runs/{run}/control")
        assert response.status_code == 404, run
        assert reason in response.json()["detail"]
    assert control.requests == []


def test_status_is_proxied(outputs, control):
    make_run(outputs, control_info=here(control.address))
    client = TestClient(server.app)
    assert client.get("/api/runs/eval-run/control").json() == STATUS


def test_steer_a_live_trace(outputs, control):
    make_run(outputs, control_info=here(control.address))
    client = TestClient(server.app)
    response = client.post("/api/runs/eval-run/control/steer", json={"text": "check the logs", "trace_id": "tr-1"})
    assert response.status_code == 200
    assert response.json()["receipts"][0]["queued"] is True
    assert control.requests == [
        {"method": "steer", "args": {"text": "check the logs", "trace_id": "tr-1", "sender": "dashboard"}}
    ]
    response = client.post(
        "/api/runs/eval-run/control/steer", json={"text": "seat only", "task_key": "job-1/solve/0", "seat": "solver"}
    )
    assert response.status_code == 200
    assert control.requests[-1]["args"] == {
        "text": "seat only",
        "task_key": "job-1/solve/0",
        "seat": "solver",
        "sender": "dashboard",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"text": " ", "trace_id": "tr-1"},
        {"text": "hi"},
        {"text": "hi", "trace_id": "tr-1", "task_key": "k"},
        {"text": "hi", "trace_id": ""},
        {"text": "hi", "episode_id": "ep-1"},
    ],
)
def test_steer_rejects_bad_requests(outputs, control, body):
    make_run(outputs, control_info=here(control.address))
    client = TestClient(server.app)
    assert client.post("/api/runs/eval-run/control/steer", json=body).status_code == 400
    assert control.requests == []


def test_run_refusal_is_a_bad_gateway(outputs, control):
    make_run(outputs, control_info=here(control.address))
    client = TestClient(server.app)
    response = client.post("/api/runs/eval-run/control/steer", json={"text": "hi", "trace_id": "gone"})
    assert response.status_code == 502
    assert "no such trace" in response.json()["detail"]


# --------------------------------------------------------- trace_id citations


def episode(episode_id: str, trace_ids: list[str], text: str) -> dict:
    return {
        "id": episode_id,
        "kind": "eval",
        "env": {"name": "prime_flow"},
        "ok": True,
        "traces": [
            {
                "id": trace_id,
                "agent": {"name": f"seat-{i}"},
                "nodes": [
                    {"parent": None, "message": {"role": "user", "content": "start"}},
                    {"parent": 0, "message": {"role": "assistant", "content": f"{text} from {trace_id}"}},
                ],
            }
            for i, trace_id in enumerate(trace_ids)
        ],
    }


def test_trace_id_citation_resolves_to_its_episode(outputs):
    run_dir = make_run(outputs, "cited")
    records = [episode("ep-a", ["tr-a0", "tr-a1"], "alpha"), episode("ep-b", ["tr-b0", "tr-b1"], "beta")]
    (run_dir / "traces.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    client = TestClient(server.app)

    # what the report view does with {"trace_id", "node", "quote", "note"}: the
    # episodes lookup by trace gives the episode, the trace is found by its id
    found = client.get("/api/runs/cited/episodes", params={"trace": "tr-b1", "limit": 2}).json()
    assert found["total"] == 1
    row = found["episodes"][0]
    assert (row["id"], row["kind"], row["line"]) == ("ep-b", "eval", 2)
    record = client.get(f"/api/runs/cited/episodes/{row['line']}").json()
    trace_index = [trace["id"] for trace in record["traces"]].index("tr-b1")
    assert trace_index == 1
    assert "beta from tr-b1" in record["traces"][trace_index]["nodes"][1]["message"]["content"]

    assert client.get("/api/runs/cited/episodes", params={"trace": "tr-zz", "limit": 2}).json()["total"] == 0
