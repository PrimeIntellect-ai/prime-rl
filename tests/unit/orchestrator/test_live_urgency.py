"""The dispatcher flushes live deltas at once only when a durable call's lookup may be
waiting on them: endings and verdicts of traces that carry a durable ref. Every other
trace's deltas, RL's rollouts among them, wait for the regular tick."""

import uuid
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.durable import call_ref

from prime_rl.orchestrator import live
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.types import InflightEpisode, Policy


def _dispatcher() -> Dispatcher:
    return Dispatcher(
        train_envs=None,
        eval_envs=None,
        train_source=None,
        eval_source=None,
        policy_clients=SimpleNamespace(),
        policy=Policy(model_name="model"),
        progress=None,
        initial_max_inflight=1,
        max_inflight_ceiling=1,
        tasks_per_minute=None,
        max_off_policy_steps=0,
        run_id="run",
        run_name=None,
    )


def _meta() -> InflightEpisode:
    return InflightEpisode(
        kind="train",
        env_name="e",
        group_id=uuid.uuid4(),
        task=vf.Task(vf.TaskData(id="t")),
        policy_version=0,
        step=0,
        client_config=None,
    )


def _open(trace_id: str, *, durable: bool) -> dict:
    labels = {"flow": {"job": "j"}}
    if durable:
        labels["call"] = {"key": "agent/0", "fingerprint": "f", "ref": call_ref(labels, "agent/0")}
    return {"trace": trace_id, "open": True, "set": {"info": {"labels": labels}}}


def _feed(dispatcher: Dispatcher, meta: InflightEpisode, delta: dict) -> None:
    durable = live.apply(meta, delta)
    dispatcher.queue_live({"delta": delta, "dispatch": {}}, durable=durable)


def test_a_plain_traces_endings_wait_for_the_tick() -> None:
    dispatcher, meta = _dispatcher(), _meta()
    _feed(dispatcher, meta, _open("plain", durable=False))
    _feed(dispatcher, meta, {"trace": "plain", "set": {"ok": False, "is_completed": True}})
    _feed(dispatcher, meta, {"trace": "plain", "checkpoints": [{"id": "c", "turns": 1, "time": 1.0}]})
    dispatcher.retire(meta, "cancelled")

    assert not dispatcher.live_urgent.is_set()


def test_a_durable_traces_endings_flush_at_once() -> None:
    for delta in (
        {"trace": "durable", "set": {"ok": True}},
        {"trace": "durable", "checkpoints": [{"id": "c", "turns": 1, "time": 1.0}]},
        {"trace": "durable", "discard": True},
    ):
        dispatcher, meta = _dispatcher(), _meta()
        _feed(dispatcher, meta, _open("durable", durable=True))
        assert not dispatcher.live_urgent.is_set()  # streaming alone is not urgent
        _feed(dispatcher, meta, delta)
        assert dispatcher.live_urgent.is_set(), delta

    dispatcher, meta = _dispatcher(), _meta()
    _feed(dispatcher, meta, _open("durable", durable=True))
    dispatcher.retire(meta, "landed", {"durable"})
    assert not dispatcher.live_urgent.is_set()  # it landed: nothing looks for its partial
    dispatcher.retire(meta, "failed")
    assert dispatcher.live_urgent.is_set()
