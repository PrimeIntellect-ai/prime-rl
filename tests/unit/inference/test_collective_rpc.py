from collections import deque
from types import SimpleNamespace

import pytest

from prime_rl.inference.patches import monkey_patch_collective_rpc_drain_failed_replies


class _Queue:
    def __init__(self, replies):
        self.replies = deque(replies)

    def dequeue(self, timeout=None):
        return self.replies.popleft()

    def enqueue(self, message):
        pass


def test_failed_rpc_drains_every_worker_reply():
    """Worker 0 fails the first RPC; worker 1's reply to it must not be returned for the second."""
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

    monkey_patch_collective_rpc_drain_failed_replies()
    ok, failed = WorkerProc.ResponseStatus.SUCCESS, WorkerProc.ResponseStatus.FAILURE
    executor = SimpleNamespace(
        rpc_broadcast_mq=_Queue([]),
        is_failed=False,
        futures_queue=deque(),
        response_mqs=[
            _Queue([(failed, "cancelled"), (ok, "init-0")]),
            _Queue([(failed, "cancelled"), (ok, "init-1")]),
        ],
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        MultiprocExecutor.collective_rpc(executor, "update_weights_from_path")
    assert MultiprocExecutor.collective_rpc(executor, "init_broadcaster") == ["init-0", "init-1"]
