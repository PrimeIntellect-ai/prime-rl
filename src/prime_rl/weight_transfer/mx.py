"""Session-scoped metadata and synchronization channels over ModelExpress."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from modelexpress import p2p_pb2
from modelexpress.client import MxClient

Role = Literal["trainer", "inference", "orchestrator"]


@dataclass
class MxChannel:
    server_url: str
    session_id: str
    model: str
    role: Role
    channel: str
    rank: int

    def __post_init__(self) -> None:
        self.client = MxClient(server_url=self.server_url)
        self.worker_id = f"{self.session_id}:{self.role}:{self.channel}:{self.rank}"
        self._source_id: str | None = None

    def _identity(self, role: Role, channel: str) -> p2p_pb2.SourceIdentity:
        return p2p_pb2.SourceIdentity(
            mx_version="0.3.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name=self.model,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            dtype="bfloat16",
            extra_parameters={
                "session_id": self.session_id,
                "role": role,
                "channel": channel,
            },
        )

    def publish(self, payload: bytes) -> str:
        worker = p2p_pb2.WorkerMetadata(worker_rank=self.rank, nixl_metadata=payload)
        self._source_id = self.client.publish_metadata(
            self._identity(self.role, self.channel),
            worker,
            self.worker_id,
        )
        return self._source_id

    def payloads(self, role: Role, channel: str) -> dict[int, bytes]:
        refs = self.client.list_sources(self._identity(role, channel)).instances
        payloads: dict[int, bytes] = {}
        for ref in refs:
            response = self.client.get_metadata(ref.mx_source_id, ref.worker_id)
            if response.found:
                payloads[response.worker.worker_rank] = response.worker.nixl_metadata
        return payloads

    def wait_for(
        self,
        role: Role,
        channel: str,
        expected: int,
        predicate: Callable[[bytes], bool],
        timeout: float,
        poll_interval: float = 0.05,
    ) -> dict[int, bytes]:
        deadline = time.monotonic() + timeout
        while True:
            matched = {rank: payload for rank, payload in self.payloads(role, channel).items() if predicate(payload)}
            if len(matched) == expected:
                return matched
            if len(matched) > expected:
                raise RuntimeError(
                    f"ModelExpress channel {role}/{channel} returned {len(matched)} matches; expected {expected}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {expected} messages on {role}/{channel}; received {len(matched)}"
                )
            time.sleep(poll_interval)
