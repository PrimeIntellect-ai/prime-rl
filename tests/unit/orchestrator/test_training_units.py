import uuid
from collections import defaultdict
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.metrics import MetricsBuilder
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import (
    EvalRollout,
    Progress,
    RAEStats,
    TrainBatchMetrics,
    TrainRollout,
    rollouts_for_logging,
)
from prime_rl.transport import TrainingSample


def _sample(prompt_len: int, completion_mask: list[bool]) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(prompt_len)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(len(completion_mask))),
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * len(completion_mask),
        completion_temperatures=[1.0] * len(completion_mask),
        env_name="unset",
    )


def _rollout(
    *,
    samples: list[TrainingSample],
    env_name: str = "debate",
    reward: float = 0.0,
    advantage: float | None = None,
    is_filtered: bool = False,
    source_rollout_id: uuid.UUID | None = None,
) -> TrainRollout:
    rollout = TrainRollout(
        raw={"reward": reward, "trajectory": []},
        env_name=env_name,
        example_id=1,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        samples=samples,
        advantage=advantage,
        is_filtered=is_filtered,
        source_rollout_id=source_rollout_id,
    )
    for sample in samples:
        sample.reward = reward
        sample.advantage = advantage
        sample.env_name = env_name
        sample.training_mode = "rl"
    TrainSink._fill_token_usage_from_samples(rollout)
    return rollout


def _sink_for(rollouts: list[TrainRollout], episodes: list[TrainRollout] | None = None) -> TrainSink:
    sink = TrainSink.__new__(TrainSink)
    sink.batch_size = len(rollouts)
    sink.token_batch_size = None
    sink.pending_batch = list(rollouts)
    sink.pending_batch_tokens = sum(TrainSink.rollout_token_count(rollout) for rollout in rollouts)
    sink.post_filters = []
    sink.pending_episode_rollouts = {episode.rollout_id: episode for episode in episodes or []}
    sink.arrivals_by_env = defaultdict(int, {"debate": len(rollouts)})
    sink.errors_by_env = defaultdict(int)
    sink.rae_step_stats = None
    return sink


def test_process_batch_ships_training_units_and_logs_source_episode():
    episode = _rollout(samples=[], reward=0.5)
    member_a_sample = _sample(prompt_len=3, completion_mask=[True, True])
    member_b_sample = _sample(prompt_len=5, completion_mask=[True])
    member_a = _rollout(
        samples=[member_a_sample],
        reward=0.8,
        advantage=0.3,
        source_rollout_id=episode.rollout_id,
    )
    member_b = _rollout(
        samples=[member_b_sample],
        reward=0.2,
        advantage=-0.1,
        is_filtered=True,
        source_rollout_id=episode.rollout_id,
    )

    batch = _sink_for([member_a, member_b], episodes=[episode]).process_batch()

    assert batch.rollouts == [member_a, member_b]
    assert batch.samples == [member_a_sample]
    assert batch.samples[0].reward == 0.8
    assert batch.samples[0].advantage == 0.3
    assert batch.samples[0].env_name == "debate"
    assert batch.samples[0].training_mode == "rl"
    assert batch.metrics.rollout_prefill_lens == [3, 5]
    assert batch.metrics.rollout_decode_lens == [2, 1]
    assert batch.metrics.samples_per_rollout == [1, 1]
    assert batch.metrics.num_prefill_tokens == 8
    assert batch.metrics.num_decode_tokens == 3
    assert batch.metrics.n_trainable == 1
    assert batch.episode_rollouts == [episode]
    assert rollouts_for_logging(batch) == [episode]


def test_rollouts_for_logging_preserves_single_agent_rollouts_in_mixed_batches():
    single_agent = _rollout(samples=[_sample(prompt_len=2, completion_mask=[True])], env_name="math", reward=1.0)
    episode = _rollout(samples=[], reward=0.5)
    member_a = _rollout(samples=[_sample(prompt_len=3, completion_mask=[True])], source_rollout_id=episode.rollout_id)
    member_b = _rollout(samples=[_sample(prompt_len=4, completion_mask=[True])], source_rollout_id=episode.rollout_id)

    batch = _sink_for([single_agent, member_a, member_b], episodes=[episode]).process_batch()

    assert rollouts_for_logging(batch) == [single_agent, episode]


def test_process_batch_tracks_pending_tokens_for_token_batching():
    first = _rollout(samples=[_sample(prompt_len=3, completion_mask=[True, True])])
    second = _rollout(samples=[_sample(prompt_len=5, completion_mask=[True])])
    third = _rollout(samples=[_sample(prompt_len=7, completion_mask=[True, True, True])])
    sink = _sink_for([first, second, third])
    sink.batch_size = None
    sink.token_batch_size = 11

    assert sink.batch_progress() == (21, 11, "tokens")
    batch = sink.process_batch()

    assert batch.rollouts == [first, second]
    assert batch.metrics.num_prefill_tokens == 8
    assert batch.metrics.num_decode_tokens == 3
    assert sink.pending_batch == [third]
    assert sink.batch_progress() == (10, 11, "tokens")


def _timing() -> dict:
    return {
        "total": 1.0,
        "setup": {"duration": 0.1},
        "generation": {"duration": 0.5},
        "model": {"duration": 0.2},
        "env": {"duration": 0.1},
        "scoring": {"duration": 0.1},
        "overhead": 0.0,
    }


def _full_rollout(
    *,
    env_name: str,
    reward: float,
    advantage: float | None = None,
    member_id: str | None = None,
    source_rollout_id: uuid.UUID | None = None,
    group_id: uuid.UUID | None = None,
    is_filtered: bool = False,
) -> TrainRollout:
    rollout = _rollout(
        samples=[_sample(prompt_len=2, completion_mask=[True])],
        env_name=env_name,
        reward=reward,
        advantage=advantage,
        is_filtered=is_filtered,
        source_rollout_id=source_rollout_id,
    )
    rollout.raw["timing"] = _timing()
    if member_id is not None:
        rollout.raw["member_id"] = member_id
    if group_id is not None:
        rollout.group_id = group_id
    return rollout


def _batch_metrics(rollouts: list[TrainRollout], rae_stats: RAEStats | None = None) -> TrainBatchMetrics:
    n = len(rollouts)
    return TrainBatchMetrics(
        n_trainable=sum(1 for r in rollouts if not r.is_filtered),
        num_prefill_tokens=2 * n,
        num_decode_tokens=n,
        rollout_prefill_lens=[2] * n,
        rollout_decode_lens=[1] * n,
        samples_per_rollout=[1] * n,
        samples_shipped=n,
        rae_stats=rae_stats,
    )


def _metrics_builder(env_group_sizes: dict[str, int]) -> MetricsBuilder:
    config = SimpleNamespace(
        train=SimpleNamespace(
            env=[SimpleNamespace(resolved_name=name, group_size=size) for name, size in env_group_sizes.items()]
        )
    )
    return MetricsBuilder(config)


def _build(builder: MetricsBuilder, rollouts: list[TrainRollout], **overrides) -> dict:
    kwargs = dict(
        step=0,
        rollouts=rollouts,
        metrics=_batch_metrics(rollouts),
        progress=Progress(),
        step_time=0.0,
        save_ckpt_time=0.0,
        teacher_logprobs_time=0.0,
        pre_filter_seen=0,
        pre_filter_dropped=0,
        pre_filter_dropped_by_name={},
        bridge_metrics={"attempts": 0, "successes": 0, "failures": 0},
    )
    kwargs.update(overrides)
    return builder.build(**kwargs)


def test_metrics_builder_emits_member_metrics_and_omits_solve_rates_for_ma_rows():
    episode_id = uuid.uuid4()
    group_id = uuid.uuid4()
    prover = _full_rollout(
        env_name="debate",
        reward=1.0,
        advantage=0.9,
        member_id="prover",
        source_rollout_id=episode_id,
        group_id=group_id,
    )
    judge = _full_rollout(
        env_name="debate",
        reward=-1.0,
        advantage=-0.9,
        member_id="judge",
        source_rollout_id=episode_id,
        group_id=group_id,
        is_filtered=True,
    )
    rollouts = [prover, judge]
    rae_stats = RAEStats(
        updates=2,
        cold_updates=1,
        baseline_abs_delta_sum=0.2,
        baseline_sum_by_member={"prover": 0.1, "judge": -0.1},
        updates_by_member={"prover": 1, "judge": 1},
        baseline_keys_total=2,
    )

    to_log = _build(
        _metrics_builder({"debate": 1}),
        rollouts,
        metrics=_batch_metrics(rollouts, rae_stats=rae_stats),
        bridge_metrics={"attempts": 4, "successes": 3, "failures": 1},
    )

    assert to_log["multi_agent/member_rows"] == 2
    assert to_log["multi_agent/fan_out_factor"] == 2.0
    assert to_log["multi_agent/prover/reward_mean"] == 1.0
    assert to_log["multi_agent/judge/reward_mean"] == -1.0
    assert to_log["multi_agent/prover/advantage_mean"] == pytest.approx(0.9)
    assert to_log["multi_agent/judge/advantage_mean"] == pytest.approx(-0.9)
    # judge row is filtered, so all trainable member rows belong to prover
    assert to_log["multi_agent/prover/trainable_row_share"] == 1.0
    assert to_log["multi_agent/judge/trainable_row_share"] == 0.0
    assert to_log["multi_agent/rae/cold_key_fraction"] == 0.5
    assert to_log["multi_agent/rae/baseline_drift_mean"] == pytest.approx(0.1)
    assert to_log["multi_agent/rae/baseline_keys_total"] == 2
    assert to_log["multi_agent/prover/rae_baseline_mean"] == pytest.approx(0.1)
    assert to_log["multi_agent/judge/rae_baseline_mean"] == pytest.approx(-0.1)
    # Zero-sum member rows have no group-solve semantics — keys omitted
    # instead of the old degenerate solve_none=1.0 / effective_batch_size=0.0
    assert "solve_none/all" not in to_log
    assert "solve_all/all" not in to_log
    assert "effective_batch_size/all" not in to_log
    assert "solve_none/debate" not in to_log
    assert to_log["bridge/attempt_count"] == 4
    assert to_log["bridge/hit_rate"] == 0.75
    assert to_log["bridge/miss_count"] == 1


def test_metrics_builder_solve_rates_ignore_member_rows_in_mixed_batches():
    episode_id = uuid.uuid4()
    ma_group = uuid.uuid4()
    sa_group = uuid.uuid4()
    rollouts = [
        _full_rollout(
            env_name="debate",
            reward=1.0,
            advantage=0.9,
            member_id="prover",
            source_rollout_id=episode_id,
            group_id=ma_group,
        ),
        _full_rollout(
            env_name="debate",
            reward=-1.0,
            advantage=-0.9,
            member_id="judge",
            source_rollout_id=episode_id,
            group_id=ma_group,
        ),
        _full_rollout(env_name="math", reward=1.0, group_id=sa_group),
        _full_rollout(env_name="math", reward=0.0, group_id=sa_group),
    ]

    to_log = _build(_metrics_builder({"debate": 1, "math": 2}), rollouts)

    # Only the math group counts: partially solved, so neither none nor all
    assert to_log["solve_none/all"] == 0.0
    assert to_log["solve_all/all"] == 0.0
    assert to_log["effective_batch_size/all"] == 1.0
    assert to_log["solve_none/math"] == 0.0
    assert "solve_none/debate" not in to_log
    # No bridge attempts in the window → no bridge keys
    assert "bridge/hit_rate" not in to_log
    assert "bridge/miss_count" not in to_log


def test_member_truncation_derives_from_member_trajectory_not_episode():
    episode = _rollout(samples=[], reward=0.0)
    episode.raw["is_truncated"] = True
    episode.raw["stop_condition"] = "prompt_too_long"

    truncated_member = {"member_id": "prover", "trajectory": [{"is_truncated": False}, {"is_truncated": True}]}
    clean_member = {"member_id": "judge", "trajectory": [{"is_truncated": False}]}
    TrainSink._inherit_episode_accounting(truncated_member, episode)
    TrainSink._inherit_episode_accounting(clean_member, episode)

    assert truncated_member["is_truncated"] is True
    # Episode truncation no longer smears onto members whose own steps are clean
    assert clean_member["is_truncated"] is False
    # Episode-level fields still inherit
    assert clean_member["stop_condition"] == "prompt_too_long"


def test_eval_sink_forwards_numeric_rollout_metric_means():
    def _eval_rollout(example_id: int, metrics: dict) -> EvalRollout:
        return EvalRollout(
            raw={
                "reward": 0.0,
                "token_usage": {"final_output_tokens": 5.0},
                "completion": "x",
                "trajectory": [],
                "metrics": metrics,
            },
            env_name="debate",
            example_id=example_id,
            group_id=uuid.uuid4(),
            policy_version=0,
            off_policy_steps=0,
        )

    rollout_a = _eval_rollout(1, {"accuracy_prover": 1.0, "judge_selected_correct": True, "note": "skip-me"})
    rollout_b = _eval_rollout(2, {"accuracy_prover": 0.0})
    sink = EvalSink.__new__(EvalSink)
    sink.eval_envs = SimpleNamespace(get=lambda name: SimpleNamespace(config=SimpleNamespace(group_size=1)))
    sink.pending_batches = {("debate", 0): [rollout_a, rollout_b]}

    batch = sink.process_batch(("debate", 0))

    # Key-agnostic means: only numeric values, averaged over carriers
    assert batch.metrics.env_metrics == {"accuracy_prover": 0.5, "judge_selected_correct": 1.0}
    wandb_dict = batch.metrics.to_wandb_dict(env_name="debate", step=0)
    assert wandb_dict["eval/debate/metrics/accuracy_prover"] == 0.5
    assert wandb_dict["eval/debate/metrics/judge_selected_correct"] == 1.0
    assert "eval/debate/metrics/note" not in wandb_dict
