"""Cross-node coordination of the eval: a key-value store on node 0, the only channel left once the trainer's process
groups are gone (the volume is no channel: hosted runs stage it per node). Node 0's eval process hosts a
``torch.distributed.TCPStore`` on ``MASTER_ADDR:coord_port``; the other nodes connect to it.

Keys (JSON values):
    node/<n>            {"ip", "host"}             every node, at start
    round/<i>           {"assign": {node: name}}   node 0: what each node serves in round i (absent: idle)
    ready/<i>/<n>       {"ok", "urls"|"error"}     every node with an assignment, once its engines answer
    stop/<i>            {}                         node 0: round i is over, stop the engines
    stopped/<i>/<n>     {}                         every node, after stopping
    done                {"ok"}                     node 0: the eval is over, clean up and exit
"""

import json
import time
from datetime import timedelta
from typing import Any, Callable


class MemoryStore:
    """A dict with the TCPStore calls the coordinator uses (tests, and a 1-node run)."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def set(self, key: str, value: bytes | str) -> None:
        self.data[key] = value.encode() if isinstance(value, str) else value

    def get(self, key: str) -> bytes:
        return self.data[key]

    def check(self, keys: list[str]) -> bool:
        return all(key in self.data for key in keys)


def tcp_store(host: str, port: int, is_master: bool, timeout: float):
    from torch.distributed import TCPStore

    return TCPStore(
        host, port, world_size=None, is_master=is_master, timeout=timedelta(seconds=timeout), wait_for_workers=False
    )


class Coordinator:
    def __init__(self, store, poll: float = 5.0) -> None:
        self.store = store
        self.poll = poll

    def set(self, key: str, value: Any) -> None:
        self.store.set(key, json.dumps(value))

    def has(self, key: str) -> bool:
        return bool(self.store.check([key]))

    def get(self, key: str) -> Any | None:
        if not self.has(key):
            return None
        return json.loads(self.store.get(key))

    def wait_any(
        self, keys: list[str], deadline: float, abort: Callable[[], bool] | None = None, clock=time.monotonic
    ) -> tuple[str, Any] | None:
        """The first of ``keys`` to exist (in list order when several do), None at the deadline or on ``abort()``.
        Connection errors count as absent keys until the deadline (node 0 restarting its store, a slow network)."""
        while True:
            for key in keys:
                try:
                    value = self.get(key)
                except Exception:
                    value = None
                if value is not None:
                    return key, value
            if clock() >= deadline or (abort is not None and abort()):
                return None
            time.sleep(self.poll)

    def wait_all(self, keys: list[str], deadline: float, clock=time.monotonic) -> dict[str, Any]:
        """Every key that exists by the deadline."""
        found: dict[str, Any] = {}
        while True:
            for key in keys:
                if key in found:
                    continue
                try:
                    value = self.get(key)
                except Exception:
                    value = None
                if value is not None:
                    found[key] = value
            if len(found) == len(keys) or clock() >= deadline:
                return found
            time.sleep(self.poll)
