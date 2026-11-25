import asyncio
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest

from prime_rl.configs.env_worker_pool import ScoringConfig
from prime_rl.env.sample_server import SampleServer
from prime_rl.env.types import SampleToScore
from prime_rl.transports.batch.types import SampleBody
from prime_rl.transports.payload import PayloadServer, RolloutStore, StoreWriter


@pytest.mark.parametrize("retained", [False, True], ids=["in-memory", "rollout-store"])
def test_a_sample_freed_while_it_is_scored_stays_freed(tmp_path, retained):
    store = RolloutStore(tmp_path / "store", "127.0.0.1") if retained else PayloadServer("127.0.0.1")
    writer = StoreWriter(store.directory, store.url) if retained else store
    body = SampleBody(
        token_ids=np.int32([1, 2]).tobytes(),
        mask=np.bool_([0, 1]).tobytes(),
        sampled=np.bool_([0, 1]).tobytes(),
        logprobs=np.float32([0, -0.5]).tobytes(),
        versions=np.int32([0, 0]).tobytes(),
    )
    writer.put("kept", msgspec.msgpack.encode(body))
    writer.put("cancelled", msgspec.msgpack.encode(body))

    async def score_sample(token_ids, sampled, context):
        if context["cancel"]:
            writer.delete("cancelled")  # the env worker's discard, as its run is cancelled
        return np.float32([-1.0, -2.0])

    server = SampleServer(store, "tcp://127.0.0.1:*", ScoringConfig())
    teacher = SimpleNamespace(score_sample=score_sample)
    try:
        asyncio.run(server.score_sample(teacher, SampleToScore(key="kept", context={"cancel": False})))
        asyncio.run(server.score_sample(teacher, SampleToScore(key="cancelled", context={"cancel": True})))
    finally:
        server.socket.close()
        server.ctx.term()

    scored = msgspec.msgpack.decode(store.get("kept"), type=SampleBody)
    assert np.frombuffer(scored.ref_logprobs, np.float32).tolist() == [-1.0, -2.0]
    assert store.get("cancelled") is None
