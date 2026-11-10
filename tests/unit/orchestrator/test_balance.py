import pytest

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.balance import (
    BalanceAdvisor,
    Capacity,
    InferencePoll,
    Kind,
    TrainerStep,
    best_split,
    inference_poll,
)
from prime_rl.orchestrator.concurrency import EngineLoadSample

POLL_S = 5.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def pools(trainer: int = 2, inference: int = 6) -> list[BalancePoolConfig]:
    return [
        BalancePoolConfig(name="trainer", units=trainer, keys=["deployment.num_train_gpus"], per_step=[1]),
        BalancePoolConfig(name="inference", units=inference, keys=["deployment.num_infer_gpus"], per_step=[1]),
    ]


def pd_pools() -> list[BalancePoolConfig]:
    return [
        BalancePoolConfig(name="trainer", units=4, keys=["deployment.num_train_nodes"], per_step=[1]),
        BalancePoolConfig(name="prefill", units=4, keys=["inference.deployment.num_prefill_replicas"], per_step=[1]),
        BalancePoolConfig(name="decode", units=4, keys=["inference.deployment.num_decode_replicas"], per_step=[1]),
    ]


def advisor(clock: Clock, pool_list: list[BalancePoolConfig] | None = None, **kwargs) -> BalanceAdvisor:
    config = BalanceConfig(pools=pool_list or pools(), window_minutes=30, sustain_minutes=10, **kwargs)
    return BalanceAdvisor(config, clock=clock)


def run(
    balance: BalanceAdvisor,
    clock: Clock,
    minutes: float,
    *,
    tokens_per_s: float,
    rollouts_per_step: int,
    step_s: float,
    busy_s: float | None,
    knee: bool,
    env_bound: bool = False,
    finished_per_s: float | None = None,
    **poll,
) -> list[str]:
    """Feed ``minutes`` of polls and steps; return the notices. The engines finish rollouts at the trainer's pace
    unless ``finished_per_s`` says otherwise."""
    notices = []
    finished_per_s = rollouts_per_step / step_s if finished_per_s is None else finished_per_s
    finished_credit = 0.0
    next_step = clock.now + step_s
    end = clock.now + minutes * 60
    while clock.now < end:
        clock.now += POLL_S
        finished_credit += finished_per_s * POLL_S
        finished = int(finished_credit)
        finished_credit -= finished
        balance.observe(
            InferencePoll(
                seconds=POLL_S,
                generated=int(tokens_per_s * POLL_S),
                finished=finished,
                knee=knee,
                env_bound=env_bound,
                **poll,
            )
        )
        if clock.now >= next_step:
            next_step += step_s
            balance.observe_step(TrainerStep(rollouts=rollouts_per_step, period=step_s, busy=busy_s))
            advice = balance.evaluate()
            if advice is not None and (notice := balance.notice(advice)) is not None:
                notices.append(notice)
    return notices


def test_best_split_balances_the_pools():
    capacities = {
        "trainer": Capacity(0.5, Kind.MEASURED, ""),
        "inference": Capacity(1.0, Kind.MEASURED, ""),
    }
    current, best = best_split(pools(), capacities)
    assert current.speed == pytest.approx(0.5)
    # 3 trainer GPUs take 0.75/s, 5 inference GPUs make 0.83/s; 4 / 4 would make 0.67/s
    assert best.units == {"trainer": 3, "inference": 5}
    assert best.speed == pytest.approx(0.75)


def test_best_split_keeps_replica_steps_and_minimums():
    pool_list = [
        BalancePoolConfig(name="trainer", units=4, step=2, minimum=2),
        BalancePoolConfig(name="inference", units=4, step=4, minimum=4),
    ]
    capacities = {"trainer": Capacity(10.0, Kind.MEASURED, ""), "inference": Capacity(1.0, Kind.MEASURED, "")}
    current, best = best_split(pool_list, capacities)
    # Inference can only move in replicas of 4: nothing smaller than 4 trainer GPUs leaves room for a second one
    assert best.units == current.units


def test_inference_bound_run_moves_gpus_to_inference():
    clock = Clock()
    balance = advisor(clock)
    # 64 rollouts per 120 s step, the trainer busy 40 s of it; the engines at their knee
    notices = run(balance, clock, 45, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=40, knee=True)
    advice = balance.advice
    assert advice.bottleneck == "inference"
    assert advice.capacities["trainer"].kind is Kind.MEASURED
    assert advice.capacities["trainer"].rate == pytest.approx(64 / 40)
    assert advice.capacities["inference"].kind is Kind.MEASURED
    assert advice.capacities["inference"].rate == pytest.approx(64 / 120, rel=0.02)
    assert advice.trainer_busy == pytest.approx(1 / 3)
    assert advice.best.units == {"trainer": 1, "inference": 7}
    assert advice.gain == pytest.approx(7 / 6, rel=0.02)
    # One notice once the split promised its gain for the sustain time, with the overrides to apply it
    assert len(notices) == 1
    assert "Inference-bound" in notices[0]
    assert "--deployment.num_train_gpus 1 --deployment.num_infer_gpus 7" in notices[0]


def test_trainer_bound_run_needs_a_knee_measurement():
    clock = Clock()
    balance = advisor(clock, probe=True)
    # The trainer is busy the whole step and inference runs below its knee: its capacity is only a lower bound, so
    # no split that takes GPUs from inference is advised
    notices = run(balance, clock, 45, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=120, knee=False)
    advice = balance.advice
    assert advice.bottleneck == "trainer"
    assert advice.capacities["inference"].kind is Kind.LOWER_BOUND
    assert not advice.changed
    assert notices == []
    assert balance.should_probe(advice)

    # The probe lifts the trainer-side bounds; the engines reach a knee at 2.5x the tokens
    balance.start_probe()
    assert not balance.should_probe(advice)
    run(
        balance,
        clock,
        6,
        tokens_per_s=2500,
        rollouts_per_step=64,
        step_s=120,
        busy_s=120,
        knee=True,
        finished_per_s=2.5 * 64 / 120,
    )
    assert balance.probe_done()
    balance.end_probe()

    # Back below the knee: the fresh knee measurement stands in for inference capacity
    notices = run(balance, clock, 40, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=120, knee=False)
    advice = balance.advice
    assert advice.capacities["inference"].kind is Kind.ESTIMATED
    # The knee made 2.5x the rollouts. Per GPU the trainer takes 0.27/s and inference makes 0.22/s, so 4 / 4 is the
    # fastest whole split: min(1.07, 0.89) = 0.89/s, 1.67x the trainer's 0.53/s
    assert advice.capacities["inference"].rate == pytest.approx(2.5 * advice.rate, rel=0.05)
    assert advice.best.units == {"trainer": 4, "inference": 4}
    assert len(notices) == 1


def test_unmeasured_inference_is_not_named_the_bottleneck():
    clock = Clock()
    balance = advisor(clock, probe=True)
    # The trainer idles 5 % of each step (e.g. payload fetches): its capacity edges past the rate, which is all
    # that is known of inference
    run(balance, clock, 45, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=114, knee=False)
    advice = balance.advice
    assert advice.capacities["trainer"].rate > advice.capacities["inference"].rate
    assert advice.bottleneck == "trainer"
    assert balance.should_probe(advice)


def test_idle_trainer_makes_unmeasured_inference_the_bottleneck():
    clock = Clock()
    balance = advisor(clock, probe=True)
    # The trainer idles two thirds of each step and the engines never reached a measured knee: inference is only
    # bounded from below, but it is the pool the run waits on
    run(balance, clock, 45, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=40, knee=False)
    advice = balance.advice
    assert advice.capacities["inference"].kind is Kind.LOWER_BOUND
    assert advice.bottleneck == "inference"
    assert not balance.should_probe(advice)
    # Moving a GPU to inference is safe: the conservative speed still gains
    assert advice.best.units == {"trainer": 1, "inference": 7}


def test_probe_gives_up_after_its_time():
    clock = Clock()
    balance = advisor(clock, probe=True, probe_max_minutes=5)
    balance.start_probe()
    run(balance, clock, 4, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=120, knee=False)
    assert not balance.probe_done()
    run(balance, clock, 2, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=120, knee=False)
    assert balance.probe_done()


def test_environment_bound_run_advises_no_split():
    clock = Clock()
    balance = advisor(clock)
    notices = run(
        balance,
        clock,
        45,
        tokens_per_s=1000,
        rollouts_per_step=64,
        step_s=120,
        busy_s=60,
        knee=False,
        env_bound=True,
    )
    assert balance.advice.bottleneck == "environment"
    assert not balance.advice.changed
    assert notices == []


def test_pd_split_follows_the_queueing_role():
    clock = Clock()
    balance = advisor(clock, pd_pools(), unit="node", infer_replicas=1)
    # Decode queues at the knee; prefill engines run a request in 40 % of polls: prefill has 2.5x slack
    notices = run(
        balance,
        clock,
        45,
        tokens_per_s=1000,
        rollouts_per_step=64,
        step_s=120,
        busy_s=60,
        knee=True,
        queued_share={"prefill": 0.0, "decode": 0.2},
        prefill_busy=0.4,
    )
    advice = balance.advice
    assert advice.bottleneck == "decode"
    assert advice.capacities["prefill"].kind is Kind.ESTIMATED
    assert advice.capacities["prefill"].rate == pytest.approx(2.5 * advice.capacities["decode"].rate)
    assert advice.best.units["decode"] > 4
    assert advice.best.units["prefill"] < 4
    assert len(notices) == 1
    assert "--inference.deployment.num_decode_replicas" in notices[0]
    per_replica = advice.best.units["prefill"] + advice.best.units["decode"]
    assert f"--deployment.num_infer_nodes {per_replica}" in notices[0]


def test_full_decode_kv_gives_decode_no_estimated_slack():
    clock = Clock()
    balance = advisor(clock, pd_pools(), unit="node")
    run(
        balance,
        clock,
        45,
        tokens_per_s=1000,
        rollouts_per_step=64,
        step_s=120,
        busy_s=120,
        knee=True,
        queued_share={"prefill": 0.3, "decode": 0.0},
        decode_kv=0.97,
    )
    assert balance.advice.capacities["decode"].kind is Kind.LOWER_BOUND
    assert balance.advice.best.units["decode"] >= 4


def test_notice_waits_for_payback():
    clock = Clock()
    balance = advisor(clock)
    run(balance, clock, 45, tokens_per_s=1000, rollouts_per_step=64, step_s=120, busy_s=40, knee=True)
    advice = balance.advice
    balance.noticed = None
    # 1 h left saves ~9 min at +17%: a 20 min restart does not pay back
    assert balance.notice(advice, remaining_s=3600, restart_s=1200) is None
    notice = balance.notice(advice, remaining_s=36000, restart_s=1200)
    assert notice is not None and "restart ~20 min" in notice


def test_inference_poll_pools_engines():
    samples = [
        EngineLoadSample("p0", "prefill", None, None, waiting=0, running=2, generation_tokens_delta=1),
        EngineLoadSample("p1", "prefill", None, None, waiting=0, running=0),
        EngineLoadSample("d0", "decode", None, None, waiting=0, generation_tokens_delta=100, kv_usage=0.5),
        EngineLoadSample("d1", "decode", None, None, waiting=0, generation_tokens_delta=50, kv_usage=0.7),
    ]
    poll = inference_poll(samples, 5.0, finished=3, knee=True, env_bound=False, queued_share={"decode": 0.1})
    assert poll.generated == 150
    assert poll.prefill_busy == 0.5
    assert poll.decode_kv == pytest.approx(0.6)


def test_rl_config_describes_the_pools():
    single = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"vllm": {"tensor_parallel_size": 2}},
            "deployment": {"type": "single_node", "num_train_gpus": 2, "num_infer_gpus": 6},
        }
    )
    balance = single.orchestrator.balance
    assert balance.unit == "GPU"
    assert [(p.name, p.units, p.step) for p in balance.pools] == [("trainer", 2, 1), ("inference", 6, 2)]

    pd = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {
                "vllm": {"tensor_parallel_size": 2},
                "deployment": {
                    "type": "disaggregated",
                    "num_prefill_replicas": 2,
                    "num_decode_replicas": 3,
                    "decode_nodes_per_replica": 2,
                },
            },
            "deployment": {"type": "multi_node", "num_train_nodes": 4},
            "slurm": {},
        }
    )
    balance = pd.orchestrator.balance
    assert balance.unit == "node"
    assert [(p.name, p.units, p.step) for p in balance.pools] == [
        ("trainer", 4, 1),
        ("prefill", 2, 1),
        ("decode", 6, 2),
    ]
