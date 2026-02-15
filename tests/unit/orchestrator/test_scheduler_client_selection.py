import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from prime_rl.orchestrator.scheduler import InflightRolloutInfo, Scheduler


def make_client(url: str) -> SimpleNamespace:
    return SimpleNamespace(api_base_url=url)


def make_scheduler(clients: list) -> Scheduler:
    pool = MagicMock()
    pool.clients = clients

    config = MagicMock()
    config.tasks_per_minute = None
    config.model.lora = None
    config.model.name = "test-model"
    config.max_inflight_rollouts = 8
    config.rollouts_per_example = 1

    return Scheduler(env=MagicMock(), inference_pool=pool, buffer=MagicMock(), config=config)


def make_task() -> MagicMock:
    task = MagicMock()
    task.done.return_value = False
    return task


def test_selects_least_loaded_client():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    # A has 3 inflight, B has 1
    for _ in range(3):
        scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_a)
    scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_b)

    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_b.api_base_url


def test_equal_counts_picks_first_client():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_a)
    scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_b)

    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url


def test_empty_inflight_picks_first_client():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url


def test_counts_update_after_completion():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    task_a = make_task()
    scheduler.inflight_rollouts[task_a] = InflightRolloutInfo(0, client_a)
    for _ in range(2):
        scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_b)

    # A=1, B=2 → selects A
    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url

    # Simulate completion of A's task
    scheduler.inflight_rollouts.pop(task_a)

    # A=0, B=2 → still selects A
    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url


def test_counts_reset_after_bulk_cancel():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_a)
    for _ in range(5):
        scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_b)

    # B heavily loaded → selects A
    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url

    scheduler.cancel_all_inflight_rollouts()

    # All cleared → both at 0, deterministic tie-break picks first (A)
    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_a.api_base_url


def test_new_client_gets_priority():
    client_a = make_client("http://a:8000/v1")
    client_b = make_client("http://b:8000/v1")
    scheduler = make_scheduler([client_a, client_b])

    for _ in range(3):
        scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_a)
    scheduler.inflight_rollouts[make_task()] = InflightRolloutInfo(0, client_b)

    # A=3, B=1 → selects B
    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_b.api_base_url

    # A third server appears with 0 inflight
    client_c = make_client("http://c:8000/v1")
    scheduler.inference_pool.clients = [client_a, client_b, client_c]

    selected = asyncio.run(scheduler._select_least_loaded_client())
    assert selected.api_base_url == client_c.api_base_url
