import pytest

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.balance import BalanceAdvisor, Capacity, InferencePoll, TrainerStep, best_split

POLL_S = 5.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def pools(trainer: int = 4, inference: int = 4) -> list[BalancePoolConfig]:
    sizes = list(range(1, trainer + inference))
    return [
        BalancePoolConfig(name="trainer", units=trainer, sizes=sizes),
        BalancePoolConfig(name="inference", units=inference, sizes=sizes),
    ]


def run(pool_list, *, trainer_busy, inference_busy, env_bound=False, paused=lambda i: False, minutes=45):
    """A run shipping 64 rollouts per 120 s step, the engines finishing them at the same pace. Returns the advisor
    and its notices."""
    clock = Clock()
    balance = BalanceAdvisor(BalanceConfig(pools=pool_list, sustain_minutes=10), clock=clock)
    notices, credit = [], 0.0
    for i in range(int(minutes * 60 / POLL_S)):
        clock.now += POLL_S
        credit += 64 / 120 * POLL_S
        finished, credit = int(credit), credit - int(credit)
        balance.observe(InferencePoll(POLL_S, finished, env_bound, dict(inference_busy), paused=paused(i)))
        if clock.now % 120 == 0:
            balance.observe_step(TrainerStep(rollouts=64, tokens=64 * 1000, period=120, busy=120 * trainer_busy))
            if (advice := balance.evaluate()) is not None and (notice := balance.notice(advice)) is not None:
                notices.append(notice)
    return balance.advice, notices


def test_best_split_keeps_to_valid_sizes():
    capacities = {"trainer": Capacity(0.5, 1.0, False), "inference": Capacity(1.0, 0.5, False)}
    # 3 trainer GPUs take 0.75/s and 5 inference GPUs make 1.25/s
    assert best_split(pools(), capacities) == (0.5, {"trainer": 5, "inference": 3}, pytest.approx(0.625))
    replicas = [
        BalancePoolConfig(name="trainer", units=4, sizes=[2, 4, 6]),
        BalancePoolConfig(name="inference", units=4, sizes=[4]),
    ]
    assert best_split(replicas, capacities)[1] == {"trainer": 4, "inference": 4}


def test_inference_bound_run_moves_gpus_to_inference():
    advice, notices = run(pools(), trainer_busy=1 / 3, inference_busy={"inference": 1.0})
    assert advice.bottleneck == "inference"
    assert advice.best == {"trainer": 2, "inference": 6}
    assert notices[0] == (
        "[balance] Inference-bound: trainer 4 -> 2, inference 4 -> 6 GPUs would be at least 50% faster"
    )


def test_trainer_bound_run_reads_idle_engines():
    # Engines busy 40% would make 2.5x the rollouts; a weight-update pause in which they idled does not count
    advice, notices = run(pools(), trainer_busy=1.0, inference_busy={"inference": 0.4}, paused=lambda i: i % 24 == 0)
    assert advice.bottleneck == "trainer"
    assert advice.capacities["inference"].rate == pytest.approx(2.5 * 64 / 120, rel=0.02)
    assert advice.best == {"trainer": 5, "inference": 3}
    assert notices


def test_saturated_or_unreported_engines_are_a_lower_bound():
    for busy in (0.9, None):
        advice, notices = run(pools(), trainer_busy=1.0, inference_busy={"inference": busy})
        assert advice.capacities["inference"].lower_bound
        assert advice.bottleneck == "trainer"


def test_pd_split_follows_each_role_busy_time():
    pd = [BalancePoolConfig(name=name, units=4, sizes=list(range(1, 12))) for name in ("trainer", "prefill", "decode")]
    advice, _ = run(pd, trainer_busy=0.5, inference_busy={"prefill": 0.4, "decode": 0.7})
    assert advice.bottleneck == "decode"
    assert advice.best["decode"] > 4 and advice.best["prefill"] < 4


def test_environment_bound_run_advises_no_split():
    advice, notices = run(pools(), trainer_busy=0.5, inference_busy={"inference": 0.4}, env_bound=True)
    assert advice.bottleneck == "environment" and advice.best == advice.current
    assert notices == []


def test_rl_config_describes_the_pools():
    def balance(**config):
        return RLConfig.model_validate({"trainer": {}, "orchestrator": {}, **config}).orchestrator.balance

    single = balance(
        inference={"vllm": {"tensor_parallel_size": 2}},
        deployment={"type": "single_node", "num_train_gpus": 4, "num_infer_gpus": 4},
        trainer={"model": {"ep": 2}},
    )
    assert single.unit == "GPU"
    # An explicit EP of 2 needs an even trainer; inference moves in TP groups of 2
    assert [(p.name, p.units, p.sizes) for p in single.pools] == [
        ("trainer", 4, [2, 4, 6]),
        ("inference", 4, [2, 4, 6]),
    ]
    pd = balance(
        inference={
            "vllm": {"tensor_parallel_size": 2},
            "deployment": {
                "type": "disaggregated",
                "num_prefill_replicas": 2,
                "num_decode_replicas": 3,
                "decode_nodes_per_replica": 2,
            },
        },
        deployment={"type": "multi_node", "num_train_nodes": 4},
        slurm={},
    )
    assert pd.unit == "node"
    assert [(p.name, p.units, p.sizes[:3]) for p in pd.pools] == [
        ("trainer", 4, [1, 2, 3]),
        ("prefill", 2, [1, 2, 3]),
        ("decode", 6, [2, 4, 6]),
    ]
