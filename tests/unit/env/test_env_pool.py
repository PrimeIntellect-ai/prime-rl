"""The env worker pool ships its env config to workers as plain data: a JSON-mode
`model_dump` and `resolve_env_config` rebuilds the env's own config class, role-typed seats
included."""

import msgpack
import pytest
from verifiers.v1 import resolve_env_config

from prime_rl.env.pool import merge_replies


@pytest.mark.parametrize(
    "data",
    [
        {"taskset": {"id": "kuhn-poker"}, "player0": {"max_turns": 3}},
        {
            "id": "best-of-n",
            "n": 2,
            "taskset": {"id": "reverse-text"},
            "agent": {"harness": {"id": "null"}, "max_turns": 2},
        },
    ],
    ids=["role-typed-seats", "bundled-env"],
)
def test_env_config_survives_the_wire(data: dict) -> None:
    config = resolve_env_config(data)
    rebuilt = resolve_env_config(config.model_dump(mode="json"))
    assert type(rebuilt) is type(config)
    assert rebuilt == config


def test_broadcast_replies_merge() -> None:
    """A `save` goes to every worker; the client gets one reply with every worker's restore points."""
    replies = [
        msgpack.packb({"success": True, "error": None, "restores": {"r1": {"trace_id": "t1"}}}),
        msgpack.packb({"success": True, "error": None, "restores": {}}),
        msgpack.packb({"success": True, "error": None, "restores": {"r2": {"trace_id": "t2"}}}),
    ]
    assert msgpack.unpackb(merge_replies(replies)) == {
        "success": True,
        "error": None,
        "restores": {"r1": {"trace_id": "t1"}, "r2": {"trace_id": "t2"}},
    }
    failed = msgpack.packb({"success": False, "error": "boom"})
    merged = msgpack.unpackb(merge_replies([replies[0], failed]))
    assert merged["success"] is False and merged["error"] == "boom"
    assert merged["restores"] == {"r1": {"trace_id": "t1"}}


@pytest.mark.asyncio
async def test_broadcast_completes_when_a_worker_exits() -> None:
    """If a worker dies mid-broadcast, the broker completes the broadcast instead of hanging,
    and the live workers' restores still reach the client."""
    import asyncio
    from unittest.mock import MagicMock

    import zmq

    from prime_rl.configs.env_worker_pool import StaticPoolConfig
    from prime_rl.env.pool import EnvWorkerPool

    pool = EnvWorkerPool(StaticPoolConfig(num_workers=2), "tcp://127.0.0.1:0", {}, None, None)
    sockets = []
    procs = []

    def fake_spawn() -> None:
        i = len(pool.workers)
        addr = f"ipc://{pool._worker_path(i)}"
        srv = pool.ctx.socket(zmq.ROUTER)
        srv.bind(addr)
        dealer = pool.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(addr)
        proc = MagicMock()
        proc.is_alive.return_value = True
        sockets.append(srv)
        procs.append(proc)
        pool.workers.append({"process": proc, "dealer": dealer, "pipe": MagicMock(), "active": 0, "index": i})
        if pool._poller is not None:
            pool._poller.register(dealer, zmq.POLLIN)

    pool._spawn_worker = fake_spawn  # type: ignore[method-assign]
    task = asyncio.create_task(pool.run())
    await asyncio.sleep(0.05)
    client = pool.ctx.socket(zmq.DEALER)
    try:
        client.connect(pool.address)
        await client.send_multipart([b"req1", b"save", msgpack.packb({"dir": "/tmp"})])
        ident0, rid0, _, _ = await sockets[0].recv_multipart()
        await sockets[0].send_multipart(
            [
                ident0,
                rid0,
                b"reply",
                msgpack.packb({"success": True, "error": None, "restores": {"r1": {"trace_id": "t1"}}}),
            ]
        )
        await sockets[1].recv_multipart()
        procs[1].is_alive.return_value = False

        rid, kind, data = await asyncio.wait_for(client.recv_multipart(), timeout=3.0)
        reply = msgpack.unpackb(data, raw=False)
        assert rid == b"req1" and kind == b"reply"
        assert reply["success"] is True
        assert reply["skipped"] == ["env worker 1 exited during the save"]
        assert reply["restores"] == {"r1": {"trace_id": "t1"}}
    finally:
        client.close()
        for s in sockets:
            s.close()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
