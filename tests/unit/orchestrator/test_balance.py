import pytest

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.balance import (
    BalanceAdvisor,
    Capacity,
    EngineSteps,
    InferencePoll,
    best_split,
    inference_poll,
)
from prime_rl.orchestrator.concurrency import EngineLoadSample
from prime_rl.orchestrator.trainer_capacity import TrainerCapacity

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


def run(pool_list, *, trainer_busy, inference_busy, env_bound=False, paused=lambda i: False, engines=None, minutes=45):
    """A run shipping 64 rollouts per 120 s step, the engines finishing them at the same pace. Returns the advisor
    and its notices."""
    clock = Clock()
    trainer = TrainerCapacity()
    balance = BalanceAdvisor(BalanceConfig(pools=pool_list, sustain_minutes=10), trainer, clock=clock)
    notices, credit = [], 0.0
    for i in range(int(minutes * 60 / POLL_S)):
        clock.now += POLL_S
        credit += 64 / 120 * POLL_S
        finished, credit = int(credit), credit - int(credit)
        if engines is not None:
            balance.observe(engines(i, finished))
        else:
            balance.observe(InferencePoll(POLL_S, finished, env_bound, dict(inference_busy), paused=paused(i)))
        if clock.now % 120 == 0:
            trainer.shipped(64, requested_at=clock.now - 120 * (1 - trainer_busy), now=clock.now)
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


def step_time(seqs: float) -> float:
    """A decode step: 20 ms + 0.1 ms per sequence + 1e-7 s per context token read, at 2k context per sequence."""
    return 0.02 + 1e-4 * seqs + 1e-7 * 2048 * seqs


def decode_run(*, max_num_seqs: int, kv_usage_per_seq: float | None):
    """Decode engines busy all the time with 32-64 sequences per step, in a trainer-bound run."""

    def poll(i: int, finished: int) -> InferencePoll:
        seqs = [32, 48, 64][i % 3]
        steps = POLL_S / step_time(seqs)
        usage = kv_usage_per_seq * seqs if kv_usage_per_seq else None
        engine = EngineSteps(steps, POLL_S, seqs * steps, 2048 * seqs * steps, 0.0, max_num_seqs, usage)
        return InferencePoll(POLL_S, finished, False, {"inference": 1.0}, engines={"inference": [engine]})

    advice, _ = run(pools(), trainer_busy=1.0, inference_busy={}, engines=poll)
    return advice.capacities["inference"]


def test_saturated_decode_engines_extrapolate_to_a_full_batch():
    capacity = decode_run(max_num_seqs=512, kv_usage_per_seq=None)
    fit = capacity.batch
    assert not capacity.lower_bound and fit.full_batch == 512
    gain = (512 / step_time(512)) / (fit.batch / step_time(fit.batch))
    assert fit.gain == pytest.approx(gain, rel=0.05)
    assert capacity.rate == pytest.approx(64 / 120 * fit.gain, rel=0.05)


def test_kv_cache_caps_the_full_batch():
    # 1% of the KV cache per sequence: KV_TARGET (60%) holds 60 sequences, well under max_num_seqs
    fit = decode_run(max_num_seqs=1024, kv_usage_per_seq=0.01).batch
    assert fit.full_batch == pytest.approx(60, rel=0.1)
    assert fit.gain == pytest.approx((60 / step_time(60)) / (fit.batch / step_time(fit.batch)), rel=0.1)


def test_lockstep_moe_ranks_are_fit_as_one_engine():
    # A DP4 MoE engine whose ranks step together, each step taking 20 ms + 0.1 ms per sequence across the group, with
    # the sequences spread unevenly (an idle rank runs dummy steps); 128 sequences per rank at most
    def poll(i: int, finished: int) -> InferencePoll:
        group_seqs = [32, 48, 64][i % 3]
        steps = POLL_S / (0.02 + 1e-4 * group_seqs)
        samples = [
            EngineLoadSample(
                engine_id=f"server/{rank}",
                role=None,
                kv_capacity_tokens=None,
                max_model_len=None,
                waiting=0,
                busy_delta=POLL_S,
                steps_delta=steps,
                decode_seqs_delta=share * group_seqs * steps,
                kv_read_delta=2048 * share * group_seqs * steps,
                prefill_delta=0.0,
                max_num_seqs=128,
                lockstep_group="server",
            )
            for rank, share in enumerate([0.5, 0.3, 0.2, 0.0])
        ]
        return inference_poll(samples, POLL_S, finished=finished, env_bound=False)

    advice, _ = run(pools(), trainer_busy=1.0, inference_busy={}, engines=poll)
    fit = advice.capacities["inference"].batch
    # The group holds 512 sequences: its throughput there over its throughput at its current batch
    assert fit.full_batch == 512
    assert fit.gain == pytest.approx((512 / (0.02 + 0.0512)) / (fit.batch / (0.02 + 1e-4 * fit.batch)), rel=0.05)
