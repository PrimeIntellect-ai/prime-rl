import asyncio
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import msgspec
import numpy as np
import psutil
import pytest

from prime_rl.configs.env_worker_pool import EnvWorkerPoolConfig, ScoringConfig
from prime_rl.entrypoints.env_worker_pool import start_sample_server, stop_sample_server
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


ENV = {"taskset": {"id": "reverse-text"}, "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}}


def test_a_sample_server_that_dies_starting_says_so():
    # Not an address of this host: the sample server cannot bind its store there.
    config = EnvWorkerPoolConfig(env=ENV, serve={"address": "tcp://203.0.113.7:5555"})
    with pytest.raises(RuntimeError, match=r"exited \(code 1\) before it started"):
        start_sample_server(config, None)


def test_a_stuck_sample_server_does_not_outlive_its_pool():
    process, death_pipe, _, _ = start_sample_server(EnvWorkerPoolConfig(env=ENV), None)
    try:
        os.kill(process.pid, signal.SIGSTOP)  # stuck: it cannot stop when its pool goes
        death_pipe.close()
        stop_sample_server(process, timeout=1)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()


@pytest.mark.slow
def test_a_pool_whose_sample_server_crashes_exits_with_an_error(tmp_path):
    address_file = tmp_path / "pool.address"
    args = ["--env.taskset.id", "reverse-text", "--env.agent.harness.id", "null"]
    args += [
        "--env.agent.runtime.type",
        "subprocess",
        "--address-file",
        str(address_file),
        "--output-dir",
        str(tmp_path),
    ]
    pool = subprocess.Popen([sys.executable, "-m", "prime_rl.entrypoints.env_worker_pool", *args])
    try:
        deadline = time.monotonic() + 120
        while not address_file.exists():
            assert pool.poll() is None and time.monotonic() < deadline, "the pool did not start"
            time.sleep(0.5)
        (sample_server,) = [
            child for child in psutil.Process(pool.pid).children() if "SampleServer" in " ".join(child.cmdline())
        ]
        sample_server.send_signal(signal.SIGKILL)
        assert pool.wait(timeout=60) != 0
    finally:
        if pool.poll() is None:
            pool.kill()
