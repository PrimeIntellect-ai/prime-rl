from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.concurrency import ConcurrencyController, EngineLoadSample

# A model served at its full 262k context with ~1.55M KV tokens, trained at seq_len 14336.
KV_CAPACITY = 1_554_623
MAX_MODEL_LEN = 262_144
SEQ_LEN = 14_336


def bootstrapped_cap(episode_budget: int | None) -> int:
    controller = ConcurrencyController(ConcurrencyConfig(), fallback_cost=SEQ_LEN, episode_budget=episode_budget)
    limits: list[int] = []
    controller.bind(set_limit=limits.append, get_inflight=lambda: 0)
    controller.observe(
        [
            EngineLoadSample(
                engine_id="engine-0",
                role=None,
                kv_capacity_tokens=KV_CAPACITY,
                max_model_len=MAX_MODEL_LEN,
                kv_usage=0.0,
                running=0,
                waiting=0,
                waiting_capacity=0,
                preemptions_delta=0,
            )
        ]
    )
    return limits[-1]


def test_start_cap_is_bounded_by_episode_budget():
    assert bootstrapped_cap(None) == KV_CAPACITY // MAX_MODEL_LEN
    assert bootstrapped_cap(SEQ_LEN) == KV_CAPACITY // SEQ_LEN


def test_episode_budget_never_raises_cost_above_engine_context():
    assert bootstrapped_cap(10 * MAX_MODEL_LEN) == KV_CAPACITY // MAX_MODEL_LEN
