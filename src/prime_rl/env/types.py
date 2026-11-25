from typing import ClassVar

from pydantic import BaseModel, Field
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.trace import RunInfo


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
    env worker to its replica's algorithm worker, with the client and model its rollout
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
    record: RecordRequest | None = None
    """Return the episode's record; None: the client persists no trace of it."""
    live: bool = True
    """Stream every change of the episode's traces; False streams only their progress (phase,
    turns, errors), without messages."""


class RunResponse(BaseResponse):
    """The end of a run: what the client knows of the episode. The worker keeps the episode
    itself and its training samples."""

    head: dict | None = None
    """The episode's `EpisodeHead`, dumped."""
    record: bytes | None = None
    """The episode's record (`Episode.to_json`), when the request asked for it."""
    index: dict | None = None
    """The record's index row source (`index_record`), with the record."""
