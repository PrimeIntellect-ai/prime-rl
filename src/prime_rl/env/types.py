from typing import ClassVar

from pydantic import BaseModel, Field
from verifiers.v1 import ClientConfig, RunInfo, SamplingConfig


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


class TrainRequest(BaseModel):
    """What a train rollout needs beyond the rollout itself: the source's algorithm, whose
    rollout-local scoring runs in the worker before it keeps the episode's samples."""

    algo: dict
    """The source's dumped `AlgoConfig`."""


class SampleToScore(BaseModel):
    key: str
    """The sample's id in the replica's store."""
    context: dict = Field(default_factory=dict)
    """What the algorithm's scoring needs beyond the sample's tokens (`Algorithm.sample_context`)."""


class ScoreRequest(BaseRequest):
    """Score kept samples with the source's algorithm (`Algorithm.score_sample`): sent by an
    env worker to its replica's sample server, with the client and model its rollout
    sampled from."""

    method: ClassVar[str] = "score"
    train: TrainRequest
    client: ClientConfig
    model: str
    samples: list[SampleToScore]


class RecordRequest(BaseModel):
    """Return the episode's record (`Episode.to_json`) for the trace monitors."""

    float_decimals: int | None = 4
    """Digits per-token float streams are rounded to; None keeps every digit."""


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself: `task_data` is the dumped
    `TaskData` the worker validates into the taskset's declared type."""

    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig
    train: TrainRequest | None = None
    """Set on train rollouts: the worker keeps their training samples. None for eval."""
    run: RunInfo | None = None
    """The run identity the worker stamps on the episode before recording it."""
    env_name: str | None = None
    """The name of the source the run belongs to, which the worker stamps on the episode."""
    group_id: str | None = None
    """The rollout group the run belongs to, which the worker stamps on the episode."""
    record: RecordRequest | None = None
    """Return the episode's record; None: the client persists no trace of it."""
    live: bool = True
    """Stream every change of the episode's traces; False streams only their progress (phase,
    turns, errors), without messages."""
    restore: str | None = None
    """Path of a restore file a `save` wrote: the run continues that rollout instead
    of starting the task. A file the worker cannot read starts the task over."""


class SaveRequest(BaseRequest):
    """Take a memory checkpoint of every rollout running on the worker, and write each
    one's restore point to `<dir>/<trace id>.restore`."""

    method: ClassVar[str] = "save"
    dir: str
    timeout: float | None = None
    """Seconds each rollout gets to take its memory checkpoint. A rollout that takes
    longer has no restore point and starts over."""


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
    orphans: list[dict] = Field(default_factory=list)
    """Unused snapshots the worker could not delete. The owner retries their cleanup."""
    skipped: list[str] = Field(default_factory=list)
    """Why a running rollout took no restore point: one reason per such rollout."""


class ReleaseRequest(BaseRequest):
    """Let the rollouts a ``save`` holds at their restore points go on."""

    method: ClassVar[str] = "release"


class DiscardRequest(BaseRequest):
    """Delete box snapshots (dumped `Snapshot`s) that nothing will restore."""

    method: ClassVar[str] = "discard"
    snapshots: list[dict]


class DiscardResponse(BaseResponse):
    deleted: list[str] = Field(default_factory=list)
    """Ids of the snapshots deleted."""


class RunResponse(BaseResponse):
    """The end of a run: what the client knows of the episode. The worker keeps the episode
    itself and its training samples."""

    head: dict | None = None
    """The episode's `EpisodeHead`, dumped."""
    record: bytes | None = None
    """The episode's record (`Episode.to_json`), when the request asked for it."""
    index: dict | None = None
    """The record's index row source (`index_record`), with the record."""
