"""Phase-2 unit tests for MxRendezvous helpers — no docker-compose required.

Direct-loads mx_rendezvous.py to bypass prime_rl.transport's heavy
__init__.py import chain.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_MOD_PATH = _REPO_ROOT / "src" / "prime_rl" / "transport" / "mx_rendezvous.py"


@pytest.fixture(scope="module")
def rdzmod():
    if "prime_rl" not in sys.modules:
        pkg = types.ModuleType("prime_rl")
        pkg.__path__ = [str(_REPO_ROOT / "src" / "prime_rl")]
        sys.modules["prime_rl"] = pkg
    if "prime_rl.transport" not in sys.modules:
        sub = types.ModuleType("prime_rl.transport")
        sub.__path__ = [str(_REPO_ROOT / "src" / "prime_rl" / "transport")]
        sys.modules["prime_rl.transport"] = sub

    spec = importlib.util.spec_from_file_location(
        "prime_rl.transport.mx_rendezvous", _MOD_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prime_rl.transport.mx_rendezvous"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _FakeInst:
    worker_id: str
    worker_rank: int
    mx_source_id: str = "fake-source"


def test_filter_same_rank_keeps_only_matching(rdzmod):
    insts = [_FakeInst("w0", 0), _FakeInst("w1", 1), _FakeInst("w2", 2), _FakeInst("w1b", 1)]
    kept = rdzmod._filter_same_rank(insts, rank=1)
    assert [i.worker_id for i in kept] == ["w1", "w1b"]


def test_freshest_per_rank_keeps_largest_updated_at(rdzmod):
    insts = [_FakeInst("w0_old", 0), _FakeInst("w0_new", 0), _FakeInst("w1_only", 1), _FakeInst("w0_mid", 0)]
    metas = {"w0_old": 100, "w0_new": 300, "w1_only": 200, "w0_mid": 200}
    kept = rdzmod._freshest_per_rank(insts, metas=metas)
    by_rank = {i.worker_rank: i.worker_id for i in kept}
    assert by_rank == {0: "w0_new", 1: "w1_only"}


def test_freshest_per_rank_handles_missing_updated_at(rdzmod):
    insts = [_FakeInst("ghost", 5), _FakeInst("known", 5)]
    metas = {"known": 1}
    kept = rdzmod._freshest_per_rank(insts, metas=metas)
    assert len(kept) == 1
    assert kept[0].worker_id == "known"


def test_freshest_per_rank_returns_lone_unknown_when_no_rival(rdzmod):
    insts = [_FakeInst("only_ghost", 7)]
    kept = rdzmod._freshest_per_rank(insts, metas={})
    assert len(kept) == 1
    assert kept[0].worker_id == "only_ghost"


def test_freshest_per_rank_sorted_by_rank(rdzmod):
    insts = [_FakeInst("w2", 2), _FakeInst("w0", 0), _FakeInst("w1", 1)]
    kept = rdzmod._freshest_per_rank(insts, metas={"w0": 1, "w1": 1, "w2": 1})
    assert [i.worker_rank for i in kept] == [0, 1, 2]


def test_publish_starts_and_close_stops_heartbeat(rdzmod, monkeypatch):
    fake_client = MagicMock()
    fake_client.publish_metadata.return_value = "mx-source-xyz"

    spawned = []

    class _FakeHB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            spawned.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(rdzmod, "HeartbeatThread", _FakeHB)

    rdz = rdzmod.MxRendezvous(client=fake_client, role="trainer", rank=2, peer_world_size=4, model_name="m")
    sid = rdz.publish(nixl_metadata=b"x", tensors=[])
    assert sid == "mx-source-xyz"
    assert len(spawned) == 1
    hb = spawned[0]
    assert hb.started
    assert hb.kwargs["worker_rank"] == 2
    assert hb.kwargs["mx_source_id"] == "mx-source-xyz"
    assert hb.kwargs["nixl_manager"] is None

    rdz.close()
    assert hb.stopped
    rdz.close()


def test_publish_skips_heartbeat_when_disabled(rdzmod, monkeypatch):
    fake_client = MagicMock()
    fake_client.publish_metadata.return_value = "sid"

    spawned = []

    class _FakeHB:
        def __init__(self, **kwargs):
            spawned.append(self)

        def start(self):
            pass

    monkeypatch.setattr(rdzmod, "HeartbeatThread", _FakeHB)
    rdz = rdzmod.MxRendezvous(
        client=fake_client, role="inference", rank=0, peer_world_size=1, model_name="m", enable_heartbeat=False
    )
    rdz.publish()
    assert spawned == []


def test_publish_swallows_heartbeat_start_failure(rdzmod, monkeypatch):
    fake_client = MagicMock()
    fake_client.publish_metadata.return_value = "sid"

    class _BrokenHB:
        def __init__(self, **kwargs):
            raise RuntimeError("can't allocate thread")

    monkeypatch.setattr(rdzmod, "HeartbeatThread", _BrokenHB)
    rdz = rdzmod.MxRendezvous(client=fake_client, role="trainer", rank=0, peer_world_size=1, model_name="m")
    sid = rdz.publish()
    assert sid == "sid"
    assert rdz._heartbeat is None


def test_collect_updated_at_returns_zero_on_failure(rdzmod):
    fake_client = MagicMock()
    fake_client.get_metadata.side_effect = RuntimeError("boom")
    rdz = rdzmod.MxRendezvous(
        client=fake_client, role="trainer", rank=0, peer_world_size=1, model_name="m", enable_heartbeat=False
    )
    out = rdz._collect_updated_at([_FakeInst("a", 0), _FakeInst("b", 1)])
    assert out == {"a": 0, "b": 0}


def test_collect_updated_at_returns_zero_on_not_found(rdzmod):
    fake_client = MagicMock()

    class _Resp:
        found = False
        worker = MagicMock(updated_at=0)

    fake_client.get_metadata.return_value = _Resp()
    rdz = rdzmod.MxRendezvous(
        client=fake_client, role="trainer", rank=0, peer_world_size=1, model_name="m", enable_heartbeat=False
    )
    out = rdz._collect_updated_at([_FakeInst("x", 0)])
    assert out == {"x": 0}


def test_collect_updated_at_returns_real_value(rdzmod):
    fake_client = MagicMock()

    class _Resp:
        found = True

        def __init__(self):
            self.worker = MagicMock(updated_at=42)

    fake_client.get_metadata.return_value = _Resp()
    rdz = rdzmod.MxRendezvous(
        client=fake_client, role="trainer", rank=0, peer_world_size=1, model_name="m", enable_heartbeat=False
    )
    out = rdz._collect_updated_at([_FakeInst("x", 0)])
    assert out == {"x": 42}
