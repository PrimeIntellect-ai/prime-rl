from typing import ClassVar

from pydantic import BaseModel, Field
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.runtimes.base import Snapshot

from prime_rl.env.delta import TraceSummary


class BaseRequest(BaseModel):
    """`method` is sent as its own route frame, not as payload data."""

    method: ClassVar[str]
    RAW_FIELDS: ClassVar[tuple[str, ...]] = ()
    """Fields packed as Python objects rather than JSON: they carry raw bytes (a
    trace's per-token arrays) that msgpack ships natively."""


class BaseResponse(BaseModel):
    success: bool = True
    error: str | None = None


class HealthRequest(BaseRequest):
    method: ClassVar[str] = "health"


class HealthResponse(BaseResponse):
    pass


class InterruptRequest(BaseRequest):
    """Interrupt every running rollout (`Rollout.interrupt`) for another process to
    continue: each checkpoints its box if it can and ends with `RolloutInterrupted`,
    and its run replies as usual, its restore point streamed with it. `timeout` bounds
    the wait for a model call that a filesystem checkpoint needs."""

    method: ClassVar[str] = "interrupt"
    timeout: float = 60.0


class InterruptResponse(BaseResponse):
    interrupted: int = 0
    """How many rollouts it interrupted."""
    checkpointed: list[str] = Field(default_factory=list)
    """The trace ids of the interrupted rollouts that took a checkpoint: their restore
    points stream once the checkpoints are restorable. The others start over."""


class SaveRequest(BaseRequest):
    """Checkpoint every running rollout while it keeps running (`Rollout.save`), for
    another process to continue from: each restore point streams with its run once
    the checkpoint is restorable. `timeout` bounds the wait for the model call that a
    filesystem checkpoint needs."""

    method: ClassVar[str] = "save"
    timeout: float = 60.0
    hold: float = 0.0
    """Hold each rollout that took a checkpoint at its model call until a `release`
    request, for this many seconds at most, so the owner records it as it is at its
    checkpoint. 0 holds none."""


class SaveResponse(BaseResponse):
    running: int = 0
    """How many rollouts were running."""
    saved: list[str] = Field(default_factory=list)
    """The trace ids of the rollouts that took a checkpoint. The others have none to
    continue from."""


class ReleaseRequest(BaseRequest):
    """Let the rollouts that the last `save` holds go on."""

    method: ClassVar[str] = "release"


class CancelRequest(BaseRequest):
    """Abort an in-flight run by its wire ``request_id``. Idempotent — a
    finished or unknown run cancels successfully with ``cancelled=False``."""

    method: ClassVar[str] = "cancel"
    request_id: str


class CancelResponse(BaseResponse):
    cancelled: bool = False
    """Whether a live run was found and aborted."""


class RestoreRequest(BaseModel):
    """A trace to continue from one of its checkpoints: the trace as a client assembled
    it from the stream (`EpisodeAssembly`) and its restore point (a `restore` delta)."""

    trace: dict
    point: dict


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself: `task_data` is the dumped
    `TaskData` the server validates into the taskset's declared type. With `restore`,
    the rollout continues that trace instead of starting one (one-seat envs)."""

    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig
    restore: RestoreRequest | None = None

    RAW_FIELDS: ClassVar[tuple[str, ...]] = ("restore",)


class DiscardRequest(BaseRequest):
    """Delete box checkpoints their owner no longer needs (a client that keeps them
    past the run, `RolloutCheckpointConfig.after = "keep"`, or restores them in another
    process)."""

    method: ClassVar[str] = "discard"
    checkpoints: list[Snapshot]


class DiscardResponse(BaseResponse):
    deleted: list[str] = Field(default_factory=list)
    """The ones deleted; the rest are still needed (the newest of a running box) or
    were never there."""


class RunResponse(BaseResponse):
    """The end of a run whose traces already streamed as deltas (`env.delta`)."""

    head: dict | None = None
    """The rollout's episode without its traces — its standing (`id`/`env`/`errors`,
    carrying episode-level errors even when no trace minted); task-specific data
    preserved in `model_extra`."""

    traces: list[TraceSummary] = Field(default_factory=list)
    """The finished traces in episode order, with the sizes the client's assembly
    must match."""
