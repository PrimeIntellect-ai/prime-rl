from typing import ClassVar

from pydantic import BaseModel, Field
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.sampling import SamplingConfig


class BaseRequest(BaseModel):
    """`method` is sent as its own route frame, not as payload data."""

    method: ClassVar[str]


class BaseResponse(BaseModel):
    success: bool = True
    error: str | None = None


class HealthRequest(BaseRequest):
    method: ClassVar[str] = "health"


class HealthResponse(BaseResponse):
    pass


class CancelRequest(BaseRequest):
    """Abort an in-flight run by its wire ``request_id``. Idempotent — a
    finished or unknown run cancels successfully with ``cancelled=False``."""

    method: ClassVar[str] = "cancel"
    request_id: str


class CancelResponse(BaseResponse):
    cancelled: bool = False
    """Whether a live run was found and aborted."""


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself: `task_data` is the dumped
    `TaskData` the worker validates into the taskset's declared type."""

    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig
    restore: str | None = None
    """Path of a restore file a `save` wrote: the run continues that rollout instead
    of starting the task. A file the worker cannot read starts the task over."""


class SaveRequest(BaseRequest):
    """Take a memory checkpoint of every rollout running on the worker, and write each
    one's restore point to `<dir>/<trace id>.restore`."""

    method: ClassVar[str] = "save"
    dir: str


class RestoreRef(BaseModel):
    """A restore file, and the box snapshot it continues from. Whoever holds the
    reference owns the snapshot and deletes it (`discard`) once nothing restores it."""

    trace_id: str
    path: str
    snapshot: dict


class SaveResponse(BaseResponse):
    restores: dict[str, RestoreRef] = Field(default_factory=dict)
    """The restore points taken, by the wire ``request_id`` of the run they continue.
    A running rollout without an entry took none and starts over."""


class DiscardRequest(BaseRequest):
    """Delete box snapshots (dumped `Snapshot`s) that nothing will restore."""

    method: ClassVar[str] = "discard"
    snapshots: list[dict]


class DiscardResponse(BaseResponse):
    deleted: list[str] = Field(default_factory=list)
    """Ids of the snapshots deleted."""


class TraceSummary(BaseModel):
    """A finished trace's identity and size, for the client's assembly check."""

    id: str
    nodes: int
    calls: int


class RunResponse(BaseResponse):
    """The end of a run whose traces already streamed as deltas (`verifiers.v1.delta`)."""

    head: dict | None = None
    """The rollout's episode without its traces — its standing (`id`/`env`/`errors`,
    carrying episode-level errors even when no trace minted); task-specific data
    preserved in `model_extra`."""

    traces: list[TraceSummary] = Field(default_factory=list)
    """The finished traces in episode order, with the sizes the client's assembly
    must match."""
