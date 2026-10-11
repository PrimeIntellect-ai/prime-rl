from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress


def test_payload_pins_hold_until_a_newer_step_is_committed():
    sink = TrainSink(None, tokenizer=None, train_envs=None, progress=Progress(), batch_size=1)
    sink.pin(10, {"a"}, 3)  # the committed resume point
    sink.pin(20, {"b"}, 5)  # saved, not committed yet: step 10 must stay resumable
    assert sink.releasable({"a", "b", "c"}) == {"c"}
    assert sink.unpin_before(20) == {"a"}
    assert sink.releasable({"b"}) == set()
    assert sink.pinned_version == 5
