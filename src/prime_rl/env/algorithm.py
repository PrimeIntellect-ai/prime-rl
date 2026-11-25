"""A pool replica's algorithm worker.

Each pool replica runs one, beside its env workers. It holds the replica's sample store:
the env workers keep their rollouts' training samples in it, and trainer ranks fetch them
from it (`prime_rl.transports.payload`). It also runs the algorithm's model-calling scoring
of kept samples (`Algorithm.score_sample`, e.g. a teacher's prefill logprobs) for all of the
replica's env workers, with one limit on the calls in flight and retries, so a failed call
costs a retry rather than the rollout that produced the sample.

An env worker asks it to score over ZMQ (`ScoreClient`): `[request_id, method, payload]`,
answered with `[request_id, payload]`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import zmq
import zmq.asyncio

from prime_rl.configs.env_worker_pool import ScoringConfig
from prime_rl.env.encoding import pack, unpack
from prime_rl.env.types import BaseResponse, SampleToScore, ScoreRequest
from prime_rl.transports.batch.types import SampleBody
from prime_rl.transports.payload import PayloadServer, RolloutStore
from prime_rl.utils.logger import get_logger
from prime_rl.utils.process import set_proc_title

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from prime_rl.orchestrator.algo import Algorithm

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(SampleBody)

MAX_RETRY_DELAY_S = 30.0


class AlgorithmWorker:
    """Scores the samples a replica's env workers kept, in the replica's store."""

    def __init__(self, store: PayloadServer, address: str, scoring: ScoringConfig) -> None:
        self.store = store
        self.scoring = scoring
        self.gate = asyncio.Semaphore(scoring.max_concurrent) if scoring.max_concurrent else None
        self.algorithms: dict[str, Algorithm] = {}
        """Train sources' algorithms by config and policy endpoint, bound to this worker."""
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(address)
        self.address = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def algorithm(self, request: ScoreRequest) -> Algorithm:
        from pydantic import TypeAdapter

        from prime_rl.configs.algorithm import AlgoConfig
        from prime_rl.orchestrator.algo import build_algorithm

        key = json.dumps([request.train.algo, request.client.model_dump(mode="json"), request.model], sort_keys=True)
        if key not in self.algorithms:
            algorithm = build_algorithm(TypeAdapter(AlgoConfig).validate_python(request.train.algo), clients=None)
            algorithm.bind(request.client, request.model)
            self.algorithms[key] = algorithm
        return self.algorithms[key]

    async def score(self, request: ScoreRequest) -> None:
        algorithm = self.algorithm(request)
        await asyncio.gather(*(self.score_sample(algorithm, sample) for sample in request.samples))

    async def score_sample(self, algorithm: Algorithm, sample: SampleToScore) -> None:
        """Score one kept sample and keep its scores in its body."""
        blob = await asyncio.to_thread(self.store.get, sample.key)
        if blob is None:
            raise KeyError(f"The replica keeps no sample {sample.key}")
        body = _decoder.decode(blob)
        token_ids = np.frombuffer(body.token_ids, np.int32)
        sampled = np.frombuffer(body.sampled, np.bool_)
        for attempt in range(1, self.scoring.attempts + 1):
            try:
                async with self.gate if self.gate is not None else contextlib.nullcontext():
                    ref_logprobs = await algorithm.score_sample(token_ids, sampled, sample.context)
                break
            except Exception as error:
                if attempt == self.scoring.attempts:
                    raise
                delay = min(2.0**attempt, MAX_RETRY_DELAY_S)
                get_logger().warning(
                    f"Scoring sample {sample.key} failed (attempt {attempt}/{self.scoring.attempts}), "
                    f"retrying in {delay:.0f}s: {type(error).__name__}: {error}"
                )
                await asyncio.sleep(delay)
        if ref_logprobs is None:
            return
        if len(ref_logprobs) != len(token_ids):
            raise ValueError(f"Scores of sample {sample.key} must align with its {len(token_ids)} tokens")
        body = msgspec.structs.replace(body, ref_logprobs=np.ascontiguousarray(ref_logprobs, np.float32).tobytes())
        await asyncio.to_thread(self.store.put, sample.key, _encoder.encode(body))

    async def handle(self, identity: bytes, request_id: bytes, method: bytes, payload: bytes) -> None:
        try:
            if method != ScoreRequest.method.encode():
                raise ValueError(f"unknown method {method.decode(errors='replace')!r}")
            await self.score(ScoreRequest.model_validate(unpack(payload)))
            response = BaseResponse()
        except Exception as error:  # a failed request is the rollout's failure, not this worker's
            get_logger().warning(f"Failed to score samples - {type(error).__name__}: {error}")
            response = BaseResponse(success=False, error=f"{type(error).__name__}: {error}")
        await self.socket.send_multipart([identity, request_id, pack(response.model_dump())])

    async def run(self) -> None:
        get_logger().info(f"Started algorithm worker store={self.store.url} address={self.address}")
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                frames = await self.socket.recv_multipart()
                if len(frames) != 4:
                    get_logger().warning(f"Dropped malformed request frames={len(frames)} expected=4")
                    continue
                task = asyncio.create_task(self.handle(*frames))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            for task in tasks:
                task.cancel()
            self.socket.close()
            self.ctx.term()


class ScoreClient:
    """An env worker's side of its replica's algorithm worker."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._socket: zmq.asyncio.Socket | None = None
        self._receiver: asyncio.Task | None = None
        self._pending: dict[bytes, asyncio.Future[bytes]] = {}

    def _connect(self) -> zmq.asyncio.Socket:
        if self._socket is None:
            self._socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self.address)
            self._receiver = asyncio.create_task(self._receive())
        return self._socket

    async def _receive(self) -> None:
        while True:
            request_id, payload = await self._socket.recv_multipart()
            future = self._pending.pop(request_id, None)
            if future is not None and not future.done():
                future.set_result(payload)

    async def score(self, request: ScoreRequest) -> None:
        """Score the request's samples; raises if the algorithm worker could not."""
        socket = self._connect()
        request_id = uuid.uuid4().hex.encode()
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await socket.send_multipart([request_id, request.method.encode(), pack(request.model_dump(mode="json"))])
            response = BaseResponse.model_validate(unpack(await future))
        finally:
            self._pending.pop(request_id, None)
        if not response.success:
            raise RuntimeError(f"Scoring failed in the algorithm worker: {response.error}")


def run_algorithm_worker(
    *,
    store_dir: Path | None,
    hosts: tuple[str, str],
    scoring: dict,
    setup: Callable[[], None],
    ready: Connection,
    death_pipe: Connection,
) -> None:
    """A pool replica's algorithm worker process: report its store's URL and its address on
    ``ready``, then serve until the pool goes (``death_pipe``)."""
    from prime_rl.env.pool import _arm_teardown

    set_proc_title("AlgorithmWorker")
    _arm_teardown(death_pipe)
    setup()
    store = RolloutStore(store_dir, *hosts) if store_dir is not None else PayloadServer(*hosts)
    address = f"ipc://{tempfile.mkdtemp(prefix='prl-algorithm-', dir='/tmp')}/socket"
    worker = AlgorithmWorker(store, address, ScoringConfig(**scoring))
    ready.send((store.url, worker.address))
    ready.close()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
