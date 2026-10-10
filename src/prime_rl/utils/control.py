"""Run control: ask a run to checkpoint or pause at the next step it trains, then follow the
request until the trainer commits it. The trainer writes a commit record under
``<output_dir>/control`` once it has saved the step a request applied to; the control API and the
launcher read them."""

import asyncio
import contextlib
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel

from prime_rl.configs.shared import ControlConfig

ControlAction = Literal["checkpoint", "pause"]


class ControlCommit(BaseModel):
    id: str
    action: ControlAction
    step: int


def get_control_dir(output_dir: Path) -> Path:
    return output_dir / "control"


def write_commit(control_dir: Path, commit: ControlCommit) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=control_dir, prefix=f"{commit.id}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(commit.model_dump_json())
    os.replace(tmp_name, control_dir / f"{commit.id}.json")


def read_commit(control_dir: Path, request_id: str) -> ControlCommit | None:
    path = control_dir / f"{request_id}.json"
    if not path.exists():
        return None
    return ControlCommit.model_validate_json(path.read_text())


def paused_since(control_dir: Path, since: float) -> bool:
    """Whether a pause was committed at or after ``since`` (a ``time.time()`` timestamp)."""
    return any(
        path.stat().st_mtime >= since and ControlCommit.model_validate_json(path.read_text()).action == "pause"
        for path in control_dir.glob("*.json")
    )


ControlState = Literal["pending", "accepted", "committed"]

MAX_WAIT_SECONDS = 60

# How long a committed pause waits for a client to read the commit before the run exits.
PAUSE_REPORT_TIMEOUT_S = 60


class ControlRequest(BaseModel):
    action: ControlAction


class ControlRecord(BaseModel):
    id: str
    action: ControlAction
    state: ControlState
    step: int | None = None
    """The step the request applies to, set once the orchestrator ships it."""


class ControlConflict(Exception):
    pass


class ControlPlane:
    """Requests submitted through the API. The orchestrator applies a pending request to the next
    batch it ships."""

    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self.records: dict[str, ControlRecord] = {}
        self.pending: ControlRecord | None = None
        self.pausing = False
        self.reported: set[str] = set()

    def submit(self, action: ControlAction) -> ControlRecord:
        if self.pausing:
            raise ControlConflict("The run is pausing")
        if self.pending is not None:
            raise ControlConflict(f"Control request {self.pending.id} is still pending")
        record = ControlRecord(id=uuid.uuid4().hex, action=action, state="pending")
        self.records[record.id] = self.pending = record
        return record

    def take(self) -> ControlRecord | None:
        record, self.pending = self.pending, None
        return record

    def accept(self, record: ControlRecord, step: int) -> None:
        record.state, record.step = "accepted", step
        self.pausing = self.pausing or record.action == "pause"

    def get(self, request_id: str) -> ControlRecord | None:
        record = self.records.get(request_id)
        if record is not None and record.state == "accepted" and read_commit(self.control_dir, request_id):
            record.state = "committed"
        return record

    def wait_reported(self, request_id: str, timeout: float) -> bool:
        """Block until a client has read the commit of ``request_id``, for a run about to exit."""
        deadline = time.monotonic() + timeout
        while request_id not in self.reported and time.monotonic() < deadline:
            time.sleep(0.5)
        return request_id in self.reported

    def report(self, request_id: str) -> ControlRecord | None:
        """``get`` for an API client, remembering which commits a client has seen."""
        record = self.get(request_id)
        if record is not None and record.state == "committed":
            self.reported.add(request_id)
        return record


def create_app(plane: ControlPlane) -> FastAPI:
    app = FastAPI(title="prime-rl control")

    @app.post("/v1/control", status_code=status.HTTP_202_ACCEPTED)
    async def submit(request: ControlRequest) -> ControlRecord:
        try:
            return plane.submit(request.action)
        except ControlConflict as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e

    @app.get("/v1/control/{request_id}")
    async def get(request_id: str, wait: float = Query(0, ge=0, le=MAX_WAIT_SECONDS)) -> ControlRecord:
        """``wait``: seconds to hold the request open until the trainer commits it."""
        record = plane.report(request_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown control request {request_id}")
        deadline = time.monotonic() + wait
        while record.state != "committed" and time.monotonic() < deadline:
            await asyncio.sleep(min(1.0, deadline - time.monotonic()))
            record = plane.report(request_id)
        return record

    return app


class ControlServer(uvicorn.Server):
    """Serves on the orchestrator's event loop; the orchestrator keeps its own signal handling."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def setup_control_server(plane: ControlPlane, config: ControlConfig) -> ControlServer:
    return ControlServer(uvicorn.Config(create_app(plane), host=config.host, port=config.port, log_level="warning"))


def start_control_server_thread(plane: ControlPlane, config: ControlConfig) -> ControlServer:
    """Serve the control API from a thread, for a trainer whose loop is not async."""
    server = setup_control_server(plane, config)
    threading.Thread(target=server.run, name="control", daemon=True).start()
    return server
