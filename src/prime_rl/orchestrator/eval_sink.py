"""Evaluation-side episode, group, and epoch assembly."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from itertools import batched
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING

import msgpack
import numpy as np

from prime_rl import monitors
from prime_rl.eval.summary import EpisodeSummary
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes, EvalMetrics, TraceRecord, effective_eval_trace
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation
from prime_rl.orchestrator.utils import episode_env_name, eval_work
from prime_rl.transports.payload import delete_payloads, trace_payload_urls

if TYPE_CHECKING:
    import verifiers.v1 as vf


@dataclass
class EvalBatch:
    env_name: str
    step: int
    group_size: int
    episodes: list[EpisodeSummary] = field(default_factory=list)
    failures: list[DispatchFailure] = field(default_factory=list)
    cancelled: int = 0
    # Annotations can include per-token advantages. Bound their memory until the epoch ships.
    annotations: SpooledTemporaryFile = field(default_factory=lambda: SpooledTemporaryFile(max_size=1024 * 1024))

    @property
    def arrived(self) -> int:
        return len(self.episodes) + len(self.failures) + self.cancelled

    def metrics(self, *, effective: bool = False) -> EvalMetrics:
        episodes = []
        records = []
        for episode in self.episodes:
            traces = [t for t in episode.traces if effective_eval_trace(t)] if effective else episode.traces
            if effective and not traces:
                continue
            episodes.append(episode)
            records.extend(TraceRecord(episode, t, sampled=False, admitted=True, cancelled=False) for t in traces)
        return EvalMetrics(episodes, records, self.group_size)

    async def log_annotations(self) -> None:
        now = time.time()
        with self.annotations as stream:
            stream.seek(0)
            reader = msgpack.Unpacker(
                stream, ext_hook=lambda code, data: np.frombuffer(data, dtype=np.float32), max_buffer_size=0
            )
            for records in batched(reader, 32):
                updates = list(records)
                for update in updates:
                    update["info"]["ship"]["time"] = now
                await monitors.log_annotations(updates)


class EvalSink:
    """Keep scalar reporting inputs after each full episode has gone through the monitors."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending: dict[tuple[str, int], EvalBatch] = {}

    def _batch(self, env_name: str, step: int) -> EvalBatch:
        key = (env_name, step)
        if key not in self.pending:
            self.pending[key] = EvalBatch(env_name, step, self.eval_envs.get(env_name).config.group_size)
        return self.pending[key]

    async def add(self, episode: vf.Episode) -> EvalBatch | None:
        delete_payloads(url for trace in episode.traces for url in trace_payload_urls(trace))
        work = eval_work(episode)
        batch = self._batch(episode_env_name(episode), work.step)

        def summarize() -> tuple[EpisodeSummary, list[vf.Episode]]:
            summary = EpisodeSummary.from_episode(episode, work.policy.start if work.policy else None)
            effective = EvalEpisodes([episode], group_size=batch.group_size).effective.vf_episodes
            for update in stamp_batch(effective, work.step):
                # Branch advantages are float32 arrays; keep them binary until the monitor consumes them.
                msgpack.pack(update, batch.annotations, default=lambda array: msgpack.ExtType(1, array.tobytes()))
            return summary, effective

        summary, effective = await asyncio.to_thread(summarize)
        if effective:
            await monitors.log(effective, work.step, "eval", "effective")
        batch.episodes.append(summary)
        return self._complete(batch)

    def fail(self, failure: DispatchFailure) -> EvalBatch | None:
        if failure.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {failure.kind} dispatch failure")
        batch = self._batch(failure.env_name, failure.step)
        batch.failures.append(failure)
        return self._complete(batch)

    def cancel(self, cancellation: GroupCancellation) -> EvalBatch | None:
        batch = self._batch(cancellation.env_name, cancellation.step)
        batch.cancelled += cancellation.count
        return self._complete(batch)

    def _complete(self, batch: EvalBatch) -> EvalBatch | None:
        if batch.arrived >= self.batch_size_for(batch.env_name):
            return self.pending.pop((batch.env_name, batch.step))
        return None

    def close(self) -> None:
        for batch in self.pending.values():
            batch.annotations.close()
        self.pending.clear()

    def batch_size_for(self, env_name: str) -> int:
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int]]:
        return [(env, step, batch.arrived, self.batch_size_for(env)) for (env, step), batch in self.pending.items()]
