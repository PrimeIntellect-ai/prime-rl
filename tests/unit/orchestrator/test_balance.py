import pytest

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.balance import BalanceAdvisor, Capacity, InferencePoll, TrainerStep, best_split

POLL_S = 5.0
TRAINED_PER_ROLLOUT = 30_000


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


def run(pool_list, *, minutes=45, trainer_busy, inference_busy, env_bound=False, **config):
    """A run shipping 64 rollouts per 120 s step, the engines finishing them at the same pace. Returns the advisor
    and its notices."""
    clock = Clock()
    balance = BalanceAdvisor(
        BalanceConfig(pools=pool_list, window_minutes=30, sustain_minutes=10, **config), clock=clock
    )
    notices = []
    credit, next_step = 0.0, 120.0
    while clock.now < minutes * 60:
        clock.now += POLL_S
        credit += 64 / 120 * POLL_S
        finished, credit = int(credit), credit - int(credit)
        balance.observe(
            InferencePoll(
                seconds=POLL_S, generated=1000, finished=finished, env_bound=env_bound, busy=dict(inference_busy)
            )
        )
        if clock.now >= next_step:
            next_step += 120
            busy = None if trainer_busy is None else 120 * trainer_busy
            balance.observe_step(TrainerStep(rollouts=64, tokens=64 * TRAINED_PER_ROLLOUT, period=120, busy=busy))
            advice = balance.evaluate()
            if advice is not None and (notice := balance.notice(advice)) is not None:
                notices.append(notice)
    return balance, notices


def test_best_split_balances_the_pools_in_whole_steps():
    capacities = {"trainer": Capacity(0.5, 1.0), "inference": Capacity(1.0, 0.5)}
    current, best = best_split(pools(), capacities)
    # 3 trainer GPUs take 0.75/s and 5 inference GPUs make 0.83/s; 4 / 4 would make 0.67/s
    assert current.speed == pytest.approx(0.5)
    assert best.units == {"trainer": 3, "inference": 5}
    stepped = [
        BalancePoolConfig(name="trainer", units=4, step=2, minimum=2),
        BalancePoolConfig(name="inference", units=4, step=4, minimum=4),
    ]
    # Inference moves only in replicas of 4: no smaller trainer leaves room for a second one
    assert best_split(stepped, {"trainer": Capacity(10.0, 1.0), "inference": Capacity(1.0, 1.0)})[1].units == {
        "trainer": 4,
        "inference": 4,
    }


def test_inference_bound_run_moves_gpus_to_inference():
    # The trainer is busy a third of each step; the engines all the time
    balance, notices = run(pools(), trainer_busy=1 / 3, inference_busy={"inference": 1.0})
    advice = balance.advice
    assert advice.bottleneck == "inference"
    assert advice.capacities["trainer"].rate == pytest.approx(3 * 64 / 120)
    assert advice.best.units == {"trainer": 1, "inference": 7}
    assert len(notices) == 1
    assert "--deployment.num_train_gpus 1 --deployment.num_infer_gpus 7" in notices[0]
    # Rates show as trained tokens per second
    assert f"{3 * 64 / 120 * TRAINED_PER_ROLLOUT / 1000:.1f}K tok/s at 2 GPUs" in notices[0]


def test_trainer_bound_run_reads_inference_headroom_from_busy_time():
    # The engines are busy 40 %: they would make 2.5x the rollouts busy all the time
    balance, notices = run(pools(), trainer_busy=1.0, inference_busy={"inference": 0.4})
    advice = balance.advice
    assert advice.bottleneck == "trainer"
    assert advice.capacities["inference"].rate == pytest.approx(2.5 * 64 / 120, rel=0.02)
    # Per GPU the trainer takes 0.27/s and inference makes 0.22/s: 4 / 4 is the fastest whole split
    assert advice.best.units == {"trainer": 4, "inference": 4}
    assert len(notices) == 1


def test_unreported_busy_time_is_a_lower_bound():
    # Engines that report no busy time count only what they make now: no split takes GPUs from them
    balance, notices = run(pools(), trainer_busy=1.0, inference_busy={"inference": None})
    advice = balance.advice
    assert advice.capacities["inference"].lower_bound
    assert advice.bottleneck == "trainer"
    assert not advice.changed
    assert notices == []


def test_pd_split_follows_each_role_busy_time():
    pd = [
        BalancePoolConfig(name="trainer", units=4, keys=["deployment.num_train_nodes"], per_step=[1]),
        BalancePoolConfig(name="prefill", units=4, keys=["inference.deployment.num_prefill_replicas"], per_step=[1]),
        BalancePoolConfig(name="decode", units=4, keys=["inference.deployment.num_decode_replicas"], per_step=[1]),
    ]
    balance, notices = run(
        pd, trainer_busy=0.5, inference_busy={"prefill": 0.4, "decode": 1.0}, unit="node", infer_replicas=1
    )
    advice = balance.advice
    assert advice.bottleneck == "decode"
    assert advice.capacities["prefill"].rate == pytest.approx(2.5 * advice.capacities["decode"].rate, rel=0.02)
    assert advice.best.units["decode"] > 4 and advice.best.units["prefill"] < 4
    assert "--inference.deployment.num_decode_replicas" in notices[0]
    per_replica = advice.best.units["prefill"] + advice.best.units["decode"]
    assert f"--deployment.num_infer_nodes {per_replica}" in notices[0]


def test_environment_bound_run_advises_no_split():
    balance, notices = run(pools(), trainer_busy=0.5, inference_busy={"inference": 0.4}, env_bound=True)
    assert balance.advice.bottleneck == "environment"
    assert not balance.advice.changed
    assert notices == []


def test_notice_waits_for_payback():
    balance, _ = run(pools(), trainer_busy=1 / 3, inference_busy={"inference": 1.0})
    balance.noticed = None
    # 1 h left saves ~9 min at +17%: a 20 min restart does not pay back
    assert balance.notice(balance.advice, remaining_s=3600, restart_s=1200) is None
    assert "restart ~20 min" in balance.notice(balance.advice, remaining_s=36000, restart_s=1200)


def test_rl_config_describes_the_pools():
    single = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"vllm": {"tensor_parallel_size": 2}},
            "deployment": {"type": "single_node", "num_train_gpus": 2, "num_infer_gpus": 6},
        }
    ).orchestrator.balance
    assert single.unit == "GPU"
    assert [(p.name, p.units, p.step) for p in single.pools] == [("trainer", 2, 1), ("inference", 6, 2)]

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
    ).orchestrator.balance
    assert pd.unit == "node"
    assert [(p.name, p.units, p.step) for p in pd.pools] == [("trainer", 4, 1), ("prefill", 2, 1), ("decode", 6, 2)]


def test_saturated_decode_engines_extrapolate_to_a_full_batch():
    from prime_rl.orchestrator.balance import EngineSteps

    clock = Clock()
    balance = BalanceAdvisor(BalanceConfig(pools=pools(), window_minutes=30, sustain_minutes=10), clock=clock)
    # Decode engines busy all the time with 32-64 sequences of 2k context: step time 20 ms + 0.1 ms per sequence +
    # 1e-7 s per context token read. KV for 512 sequences: a full batch makes ~3.4x the tokens per second.
    batch = [32, 48, 64]
    for i in range(45 * 12):
        clock.now += POLL_S
        seqs = batch[i % 3]
        step = 0.02 + 1e-4 * seqs + 1e-7 * 2048 * seqs
        steps = POLL_S / step
        engine = EngineSteps(
            steps=steps,
            busy=POLL_S,
            decode_seqs=seqs * steps,
            kv_read=2048 * seqs * steps,
            prefill=0.0,
            max_num_seqs=1024,
            kv_capacity=int(512 * 2048 / 0.95),
        )
        balance.observe(
            InferencePoll(
                seconds=POLL_S,
                generated=int(seqs * steps),
                finished=1,
                env_bound=False,
                busy={"inference": 1.0},
                engines={"inference": [engine]},
            )
        )
        if i % 24 == 23:
            balance.observe_step(TrainerStep(rollouts=24, tokens=24 * TRAINED_PER_ROLLOUT, period=120, busy=120))
    advice = balance.evaluate()
    fit = advice.capacities["inference"].batch
    assert fit.full_batch == pytest.approx(512, rel=0.01)
    full = 512 / (0.02 + 1e-4 * 512 + 1e-7 * 2048 * 512)
    now = 48 / (0.02 + 1e-4 * 48 + 1e-7 * 2048 * 48)
    assert fit.gain == pytest.approx(full / now, rel=0.05)
    assert advice.capacities["inference"].rate > 2 * advice.rate


def test_polls_without_busy_time_do_not_blank_the_window():
    clock = Clock()
    balance = BalanceAdvisor(BalanceConfig(pools=pools(), window_minutes=30), clock=clock)
    for i in range(45 * 12):
        clock.now += POLL_S
        # One poll in ten misses an engine's busy time (a failed fetch, an engine that has not stepped yet)
        busy = None if i % 10 == 0 else 0.4
        balance.observe(
            InferencePoll(seconds=POLL_S, generated=1000, finished=1, env_bound=False, busy={"inference": busy})
        )
    assert balance.utilization("inference") == pytest.approx(0.4)
