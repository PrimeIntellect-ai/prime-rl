from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import RolloutDispatcher


def make_dispatcher(*, train: int, eval: int) -> RolloutDispatcher:
    dispatcher = object.__new__(RolloutDispatcher)
    dispatcher.max_train_inflight = 128
    dispatcher.max_eval_inflight = 512
    dispatcher.max_inflight = 512
    dispatcher.inflight_permits = train + eval
    dispatcher.inflight = {
        **{f"train-{i}": SimpleNamespace(kind="train", rollout_count=1) for i in range(train)},
        **{f"eval-{i}": SimpleNamespace(kind="eval", rollout_count=1) for i in range(eval)},
    }
    return dispatcher


def test_eval_has_four_times_train_capacity():
    dispatcher = make_dispatcher(train=0, eval=128)
    assert dispatcher.available_permits("eval") == 384
    assert dispatcher.available_permits("train") == 128


def test_combined_capacity_is_never_exceeded_during_eval_tail():
    dispatcher = make_dispatcher(train=0, eval=512)
    assert dispatcher.available_permits("train") == 0

    dispatcher = make_dispatcher(train=64, eval=448)
    assert dispatcher.available_permits("train") == 0
    assert dispatcher.available_permits("eval") == 0


def test_training_keeps_its_smaller_per_kind_limit():
    dispatcher = make_dispatcher(train=128, eval=0)
    assert dispatcher.available_permits("train") == 0
    assert dispatcher.available_permits("eval") == 384
